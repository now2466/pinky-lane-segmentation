"""Save a semantic class-ID mask and foreground-only color overlay."""

import argparse
import json
from pathlib import Path

import cv2

from .checkpoints import load_model, sha256
from .inference import overlay, predict, write_image
from .models import CLASSES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("checkpoint", "torchscript"), default="checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ignore-top", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output directory already exists")
    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Cannot read image")
    model, device, count, ignore_top = load_model(args.model, args.device, args.format, args.ignore_top)
    mask = predict(model, [bgr], device, ignore_top)[0]
    args.output.mkdir(parents=True)
    write_image(args.output / "mask.png", mask)
    write_image(args.output / "overlay.png", overlay(bgr, mask))
    report = {"classes": list(CLASSES[:count]), "mask_shape": list(mask.shape),
              "input_shape": [1, 3, 240, 320], "ignore_top": ignore_top,
              "model_sha256": sha256(args.model),
              "class_pixels": {name: int((mask == i).sum()) for i, name in enumerate(CLASSES[:count])}}
    (args.output / "prediction.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
