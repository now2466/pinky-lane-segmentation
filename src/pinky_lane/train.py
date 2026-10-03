"""Train only when explicitly invoked; dry-run validates data and forward pass."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .augmentations import validate
from .checkpoints import select_device, warm_start
from .dataset import LaneDataset, check_dataset
from .losses import class_weights, segmentation_loss
from .metrics import evaluate, selection_score
from .models import ARCHITECTURES, CLASSES, LaneUNet


def run(config, dataset_root, output=None, warm_start_path=None, device_name=None,
        epochs=None, dry_run=False):
    count = config["num_classes"]
    if count not in (4, 5):
        raise ValueError("num_classes must be 4 or 5")
    training = config["training"]
    augmentation = config.get("augmentation", {})
    validate(augmentation)
    batch_size = int(training["batch_size"])
    epoch_count = int(epochs if epochs is not None else training["epochs"])
    workers = int(training.get("workers", 0))
    if batch_size < 1 or epoch_count < 1 or workers < 0:
        raise ValueError("Batch size/epochs must be positive and workers nonnegative")
    ignore_top = config.get("inference", {}).get("ignore_top")
    if isinstance(ignore_top, bool) or not isinstance(ignore_top, int) or not 0 <= ignore_top < 240:
        raise ValueError("Config requires inference.ignore_top integer in [0,239]")
    if not dry_run and (output is None or Path(output).exists()):
        raise ValueError("A new output directory is required; existing paths are never overwritten")
    device = select_device(device_name or training.get("device", "auto"))
    torch.set_num_threads(int(training.get("cpu_threads", 4)))
    cv2_threads = 0
    import cv2
    cv2.setNumThreads(cv2_threads)
    seed = int(training.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    audit = check_dataset(dataset_root, count)
    train_set = LaneDataset(dataset_root, "train", count, augmentation)
    val_set = LaneDataset(dataset_root, "val", count)
    pixel_counts = np.zeros(count, dtype=np.int64)
    frame_weights = []
    multiplier = float(training.get("crosswalk_sample_multiplier", 1))
    if not np.isfinite(multiplier) or multiplier < 1:
        raise ValueError("crosswalk_sample_multiplier must be finite and >=1")
    for row in train_set.rows:
        _, mask, _ = train_set.read(row)
        pixel_counts += np.bincount(mask[mask != 255].ravel(), minlength=count)
        frame_weights.append(multiplier if np.any(mask == 3) else 1.0)
    weights = (torch.tensor(class_weights(pixel_counts), device=device) if count == 4 else None)
    # Fail before creating run output if validation cannot select a five-class checkpoint.
    if count == 5 and not any(row.get("supervision_policy") == "fully_labeled_5class"
                              for row in val_set.rows):
        raise ValueError("Five-class validation requires fully labeled five-class samples")
    model = LaneUNet(count).to(device)
    warm_hash = warm_start(model, warm_start_path) if warm_start_path else None
    if dry_run:
        model.eval()
        image, labels, policy, _ = train_set[0]
        with torch.inference_mode():
            logits = model(image.unsqueeze(0).to(device))
            loss = segmentation_loss(logits, labels.unsqueeze(0).to(device), [policy], weights)
        if not torch.isfinite(loss):
            raise RuntimeError("Dry-run produced a non-finite loss")
        return {"training_started": False, "audit": audit, "logits_shape": list(logits.shape),
                "sample_loss_finite": bool(torch.isfinite(loss)), "warm_start_sha256": warm_hash}
    output = Path(output)
    loader_options = dict(batch_size=batch_size, num_workers=workers,
                          pin_memory=device.type == "cuda")
    sampler = None
    if multiplier > 1:
        sampler = WeightedRandomSampler(frame_weights, len(frame_weights), replacement=True,
                                        generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_set, shuffle=sampler is None, sampler=sampler, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]),
                                  weight_decay=float(training.get("weight_decay", 0.0001)))
    amp = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    output.mkdir(parents=True)
    snapshot = {**config, "training": {**training, "epochs": epoch_count, "device": str(device)}}
    (output / "config.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    provenance = {"dataset_manifest_sha256": audit["manifest_sha256"],
                  "counts": audit["counts"], "num_classes": count, "seed": seed,
                  "warm_start_sha256": warm_hash,
                  "torch_version": str(torch.__version__), "device": str(device),
                  "train_pixel_counts": pixel_counts.tolist(),
                  "class_weights": weights.cpu().tolist() if weights is not None else None,
                  "source_groups_spanning_splits": audit["source_groups_spanning_splits"],
                  "selection_metric": "fully_labeled_source_macro_foreground_miou"
                  if count == 5 else "foreground_miou",
                  "metrics_absent_class_policy": "null IoU; excluded from mean if union is zero"}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    best = -1.0
    elapsed_epochs = []
    for epoch in range(1, epoch_count + 1):
        started = time.monotonic()
        model.train()
        total_loss = 0.0
        for images, labels, policies, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=amp):
                loss = segmentation_loss(model(images), labels, policies, weights,
                                         float(training.get("dice_weight", 0.3)))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(images)
        validation = evaluate(model, val_loader, device, count)
        score = selection_score(validation, count)
        duration = time.monotonic() - started
        elapsed_epochs.append(duration)
        record = {"epoch": epoch, "train_loss": total_loss / len(train_set),
                  "selection_score": score, "val": validation, "seconds": duration,
                  "estimated_remaining_seconds": float(np.mean(elapsed_epochs)) * (epoch_count-epoch)}
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        payload = {"architecture": ARCHITECTURES[count], "classes": CLASSES[:count],
                   "model_state": model.state_dict(), "epoch": epoch, "val": validation,
                   "selection_score": score, "dataset_manifest_sha256": audit["manifest_sha256"],
                   "inference": {"ignore_top": ignore_top}}
        torch.save(payload, output / "last.pt")
        if score > best:
            best = score
            torch.save(payload, output / "best.pt")
        print(json.dumps(record), flush=True)
    return {"training_started": True, "epochs_completed": epoch_count, "best_score": best}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    print(json.dumps(run(config, args.dataset, args.output, args.warm_start,
                         args.device, args.epochs, args.dry_run), indent=2))


if __name__ == "__main__":
    main()
