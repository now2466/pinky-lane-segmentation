"""Explicit validation/test evaluation, separate from checkpoint selection."""

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from .checkpoints import load_model, sha256
from .dataset import LaneDataset
from .metrics import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Evaluation output already exists")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    # Metrics use the ground-truth 255 ignore mask, not an inference display ROI.
    model, device, count, _ = load_model(args.model, args.device, ignore_top=0)
    dataset = LaneDataset(args.dataset, args.split, count)
    report = evaluate(model, DataLoader(dataset, batch_size=args.batch_size), device, count)
    report.update(split=args.split, model_sha256=sha256(args.model),
                  dataset_manifest_sha256=sha256(args.dataset / "manifest.json"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
