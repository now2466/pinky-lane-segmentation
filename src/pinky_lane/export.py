"""Export a checked TorchScript model with preprocessing/class metadata."""

import argparse
import json
from pathlib import Path

import torch

from .checkpoints import load_model, sha256
from .models import CLASSES


def export_model(model_path, output, ignore_top=None):
    output = Path(output)
    sidecar = Path(str(output) + ".json")
    if output.exists() or sidecar.exists():
        raise ValueError("Export output or metadata already exists")
    torch.set_num_threads(4)
    model, _, count, ignore_top = load_model(model_path, "cpu", ignore_top=ignore_top)
    torch.manual_seed(17)
    sample = torch.rand(1, 3, 240, 320)
    with torch.inference_mode():
        reference = model(sample)
        traced = torch.jit.trace(model, sample)
        error = float((reference-traced(sample)).abs().max())
        if error > 1e-4:
            raise RuntimeError("TorchScript export differs from eager model")
    output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output))
    loaded = torch.jit.load(str(output)).eval()
    with torch.inference_mode():
        reload_error = float((reference-loaded(sample)).abs().max())
    if reload_error > 1e-4:
        raise RuntimeError("Reloaded TorchScript differs from eager model")
    metadata = {"classes": list(CLASSES[:count]), "input_shape": [1, 3, 240, 320],
                "input_color": "RGB", "input_range": [0, 1], "output": "logits",
                "output_shape": [1, count, 240, 320], "inference": {"ignore_top": ignore_top},
                "checkpoint_sha256": sha256(model_path), "torchscript_sha256": sha256(output),
                "torch_version": str(torch.__version__), "trace_max_abs_error": error,
                "reload_max_abs_error": reload_error}
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ignore-top", type=int)
    args = parser.parse_args()
    print(json.dumps(export_model(args.model, args.output, args.ignore_top), indent=2))


if __name__ == "__main__":
    main()
