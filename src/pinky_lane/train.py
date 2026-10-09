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
from .checkpoints import load_model, select_device, warm_start
from .dataset import CONFIRMED_BACKGROUND, LaneDataset, check_dataset
from .losses import bump_positive_loss, class_weights, lane_distillation_loss, segmentation_loss
from .metrics import evaluate, selection_score
from .models import ARCHITECTURES, CLASSES, LaneUNet
from .sampling import CompleteRepeatSampler


def run(config, dataset_root, output=None, warm_start_path=None, device_name=None,
        epochs=None, dry_run=False):
    count = config["num_classes"]
    if count not in (4, 5):
        raise ValueError("num_classes must be 4 or 5")
    training = config["training"]
    small_repeat = training.get("small_bump_repeat_factor", 1)
    negative_repeat = training.get("confirmed_background_repeat_factor", 1)
    small_max = training.get("small_bump_max_pixels", 512)
    if (any(isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 10
            for n in (small_repeat, negative_repeat))
            or isinstance(small_max, bool) or not isinstance(small_max, int) or small_max < 1):
        raise ValueError("Invalid rare-case repeat configuration")
    distillation_weight = float(training.get("lane_distillation_weight", 0))
    if not np.isfinite(distillation_weight) or distillation_weight < 0:
        raise ValueError("Invalid lane_distillation_weight")
    if distillation_weight and warm_start_path is None:
        raise ValueError("Lane distillation requires warm-start teacher")
    positive_weight = float(training.get("bump_positive_weight", 0))
    positive_small_weight = float(training.get("bump_positive_small_weight", 2))
    if (not np.isfinite(positive_weight) or positive_weight < 0
            or not np.isfinite(positive_small_weight) or positive_small_weight < 1
            or (positive_weight and count != 5)):
        raise ValueError("Invalid bump_positive_weight settings")
    save_trained = training.get("save_best_trained", False)
    if not isinstance(save_trained, bool):
        raise ValueError("save_best_trained must be boolean")
    selection_mode = training.get("selection_metric", "legacy_source_macro")
    if selection_mode not in {"legacy_source_macro", "fixed_foreground"}:
        raise ValueError("Unknown checkpoint selection metric")
    confirmed_dice = training.get("confirmed_background_dice", True)
    if not isinstance(confirmed_dice, bool):
        raise ValueError("confirmed_background_dice must be boolean")
    patience = int(training.get("early_stopping_patience", 0))
    min_delta = float(training.get("min_delta", 0))
    guard = training.get("baseline_guard_max_iou_drop")
    if (patience < 0 or not np.isfinite(min_delta) or min_delta < 0
            or (guard is not None and (not np.isfinite(guard) or not 0 <= guard <= 1))):
        raise ValueError("Invalid early stopping or baseline guard")
    if guard is not None and (warm_start_path is None or selection_mode != "fixed_foreground"):
        raise ValueError("Baseline guard requires warm-start and fixed_foreground selection")
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
    repeat_counts = []
    multiplier = float(training.get("crosswalk_sample_multiplier", 1))
    if not np.isfinite(multiplier) or multiplier < 1:
        raise ValueError("crosswalk_sample_multiplier must be finite and >=1")
    for row in train_set.rows:
        _, mask, _ = train_set.read(row)
        mask = np.where(mask == CONFIRMED_BACKGROUND, 0, mask)
        pixel_counts += np.bincount(mask[mask != 255].ravel(), minlength=count)
        frame_weights.append(multiplier if np.any(mask == 3) else 1.0)
        bump_area = int(np.count_nonzero(mask == 4))
        repeats = small_repeat if 0 < bump_area <= small_max else 1
        if row.get("confirmed_background_polygons"):
            repeats = max(repeats, negative_repeat)
        repeat_counts.append(repeats)
    weights = (torch.tensor(class_weights(pixel_counts), device=device) if count == 4 else None)
    # Fail before creating run output if validation cannot select a five-class checkpoint.
    if count == 5 and not any(row.get("supervision_policy") == "fully_labeled_5class"
                              for row in val_set.rows):
        raise ValueError("Five-class validation requires fully labeled five-class samples")
    model = LaneUNet(count).to(device)
    warm_hash = warm_start(model, warm_start_path) if warm_start_path else None
    trainable_modules = training.get("trainable_modules")
    if trainable_modules is not None:
        if (not isinstance(trainable_modules, list) or not trainable_modules
                or not set(trainable_modules) <= set(model._modules)):
            raise ValueError("Invalid trainable_modules list")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.split(".")[0] in trainable_modules)
    teacher = None
    if distillation_weight:
        teacher, _, teacher_count, _ = load_model(warm_start_path, str(device), ignore_top=0)
        if teacher_count != count:
            raise ValueError("Distillation requires equal teacher/student class count")
        teacher.requires_grad_(False)
    if dry_run:
        model.eval()
        image, labels, policy, _ = train_set[0]
        with torch.inference_mode():
            logits = model(image.unsqueeze(0).to(device))
            loss = segmentation_loss(logits, labels.unsqueeze(0).to(device), [policy], weights,
                                     float(training.get("dice_weight", 0.3)), confirmed_dice)
        if not torch.isfinite(loss):
            raise RuntimeError("Dry-run produced a non-finite loss")
        return {"training_started": False, "audit": audit, "logits_shape": list(logits.shape),
                "sample_loss_finite": bool(torch.isfinite(loss)), "warm_start_sha256": warm_hash}
    output = Path(output)
    loader_options = dict(batch_size=batch_size, num_workers=workers,
                          pin_memory=device.type == "cuda")
    sampler = None
    if max(repeat_counts) > 1 and multiplier > 1:
        raise ValueError("Rare-case repeats cannot be combined with weighted replacement sampling")
    if max(repeat_counts) > 1:
        sampler = CompleteRepeatSampler(repeat_counts, seed)
    if multiplier > 1:
        sampler = WeightedRandomSampler(frame_weights, len(frame_weights), replacement=True,
                                        generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_set, shuffle=sampler is None, sampler=sampler, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)
    epoch_samples = len(sampler) if sampler is not None else len(train_set)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(training["learning_rate"]),
                                  weight_decay=float(training.get("weight_decay", 0.0001)))
    scheduler = None
    if training.get("lr_scheduler") == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5,
            patience=int(training.get("lr_scheduler_patience", 3)),
            min_lr=float(training.get("min_learning_rate", 0.000001)))
    elif training.get("lr_scheduler") not in {None, "none"}:
        raise ValueError("Unknown learning-rate scheduler")
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
                  "selection_metric": selection_mode,
                  "confirmed_background_dice": confirmed_dice,
                  "lane_distillation_weight": distillation_weight,
                  "bump_positive_weight": positive_weight,
                  "bump_positive_small_weight": positive_small_weight,
                  "save_best_trained": save_trained,
                  "trainable_modules": trainable_modules,
                  "epoch_training_samples": epoch_samples,
                  "small_bump_repeat_factor": small_repeat,
                  "small_bump_max_pixels": small_max,
                  "confirmed_background_repeat_factor": negative_repeat,
                  "metrics_absent_class_policy": "null IoU; excluded from mean if union is zero"}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    best = -1.0
    stale_epochs = 0
    baseline = None
    baseline_classes = {}
    if guard is not None:
        baseline = evaluate(model, val_loader, device, count)
        best = selection_score(baseline, count, selection_mode)
        baseline_classes = baseline["fixed_foreground"]["classes"]
        if scheduler:
            scheduler.step(best)
        (output / "baseline-val.json").write_text(json.dumps(baseline, indent=2) + "\n")
        torch.save({"architecture": ARCHITECTURES[count], "classes": CLASSES[:count],
                    "model_state": model.state_dict(), "epoch": 0, "val": baseline,
                    "selection_score": best, "selection_metric": selection_mode,
                    "dataset_manifest_sha256": audit["manifest_sha256"],
                    "warm_start_sha256": warm_hash,
                    "inference": {"ignore_top": ignore_top}}, output / "best.pt")
    best_trained = -1.0
    best_experimental = -1.0
    elapsed_epochs = []
    for epoch in range(1, epoch_count + 1):
        started = time.monotonic()
        model.train()
        if trainable_modules is not None:
            # Frozen BatchNorm statistics must not drift during head-only training.
            for name, module in model.named_children():
                if name not in trainable_modules:
                    module.eval()
        total_loss = 0.0
        for images, labels, policies, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=amp):
                logits = model(images)
                loss = segmentation_loss(logits, labels, policies, weights,
                                         float(training.get("dice_weight", 0.3)), confirmed_dice)
                if positive_weight:
                    loss = loss + positive_weight * bump_positive_loss(
                        logits, labels, policies, small_max, positive_small_weight)
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits = teacher(images)
                    loss = loss + distillation_weight * lane_distillation_loss(logits, teacher_logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(images)
        validation = evaluate(model, val_loader, device, count)
        score = selection_score(validation, count, selection_mode)
        regressions = []
        if baseline is not None:
            candidate_classes = validation["fixed_foreground"]["classes"]
            for name, values in baseline_classes.items():
                if values["gt_pixels"] and candidate_classes[name]["iou"] < values["iou"] - guard:
                    regressions.append(name)
        accepted = not regressions and score > best + min_delta
        used_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step(score)
        duration = time.monotonic() - started
        elapsed_epochs.append(duration)
        record = {"epoch": epoch, "train_loss": total_loss / epoch_samples,
                  "epoch_training_samples": epoch_samples,
                  "selection_score": score, "val": validation, "seconds": duration,
                  "learning_rate": used_lr, "next_learning_rate": optimizer.param_groups[0]["lr"],
                  "selection_metric": selection_mode, "checkpoint_accepted": accepted,
                  "baseline_guard_regressions": regressions,
                  "estimated_remaining_seconds": float(np.mean(elapsed_epochs)) * (epoch_count-epoch)}
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        payload = {"architecture": ARCHITECTURES[count], "classes": CLASSES[:count],
                   "model_state": model.state_dict(), "epoch": epoch, "val": validation,
                   "selection_score": score, "dataset_manifest_sha256": audit["manifest_sha256"],
                   "selection_metric": selection_mode,
                   "inference": {"ignore_top": ignore_top}}
        torch.save(payload, output / "last.pt")
        if save_trained and score > best_experimental:
            torch.save({**payload, "baseline_guard_regressions": regressions}, output / "best-experimental.pt")
            best_experimental = score
        if save_trained and not regressions and score > best_trained:
            torch.save(payload, output / "best-trained.pt")
            best_trained = score
        if accepted:
            best = score
            torch.save(payload, output / "best.pt")
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(json.dumps(record), flush=True)
        if patience and stale_epochs >= patience:
            break
    result = {"training_started": True, "epochs_completed": epoch, "best_score": best,
              "early_stopped": epoch < epoch_count, "selection_metric": selection_mode,
              "best_trained_score": best_trained if save_trained and best_trained >= 0 else None,
              "best_experimental_score": best_experimental if save_trained and best_experimental >= 0 else None}
    (output / "training-result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


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
