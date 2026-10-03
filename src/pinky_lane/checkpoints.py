"""Load state dictionaries safely and validate the inference contract."""

import hashlib
from pathlib import Path

import torch

from .models import ARCHITECTURES, CLASSES, LaneUNet


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices: auto, cpu, cuda, cuda:N")
    return device


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state"), dict):
        raise ValueError("Expected a checkpoint dictionary with model_state")
    state = checkpoint["model_state"]
    if "head.weight" not in state:
        raise ValueError("Missing LaneUNet output head")
    count = state["head.weight"].shape[0]
    if count not in ARCHITECTURES or checkpoint.get("architecture") != ARCHITECTURES[count]:
        raise ValueError("Checkpoint architecture and output head must agree")
    if tuple(checkpoint.get("classes", ())) != CLASSES[:count]:
        raise ValueError("Checkpoint class order does not match LaneUNet")
    return checkpoint, count


def load_model(path, device="auto", model_format="checkpoint", ignore_top=None):
    device = select_device(device)
    if model_format == "torchscript":
        model = torch.jit.load(str(path), map_location=device).eval()
        import json
        sidecar = Path(str(path) + ".json")
        metadata = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
        with torch.inference_mode():
            shape = model(torch.zeros(1, 3, 240, 320, device=device)).shape
        if len(shape) != 4 or shape[1] not in (4, 5) or tuple(shape[2:]) != (240, 320):
            raise ValueError("TorchScript must output [N,4|5,240,320] logits")
        count = shape[1]
        if "classes" in metadata and tuple(metadata["classes"]) != CLASSES[:count]:
            raise ValueError("TorchScript metadata class order mismatch")
    elif model_format == "checkpoint":
        metadata, count = load_checkpoint(path)
        model = LaneUNet(count)
        model.load_state_dict(metadata["model_state"], strict=True)
        model = model.to(device).eval()
    else:
        raise ValueError("model_format must be checkpoint or torchscript")
    # Legacy four-class weights were supervised below row 110. Old five-class
    # checkpoints do not specify a single ROI: require an explicit override.
    if ignore_top is None:
        ignore_top = metadata.get("inference", {}).get("ignore_top")
    if ignore_top is None and count == 4:
        ignore_top = 110
    if ignore_top is None:
        raise ValueError("Five-class model requires --ignore-top or inference metadata")
    if isinstance(ignore_top, bool) or not isinstance(ignore_top, int) or not 0 <= ignore_top < 240:
        raise ValueError("ignore_top must be an integer in [0,239]")
    return model, device, count, ignore_top


def warm_start(model, path):
    checkpoint, source_count = load_checkpoint(path)
    source = checkpoint["model_state"]
    target = model.state_dict()
    target_count = target["head.weight"].shape[0]
    if source_count == target_count:
        model.load_state_dict(source, strict=True)
    elif source_count == 4 and target_count == 5:
        if set(source) != set(target):
            raise ValueError("Checkpoint parameter keys do not match")
        for key, value in source.items():
            if key in {"head.weight", "head.bias"}:
                target[key][:4].copy_(value)
            else:
                if target[key].shape != value.shape:
                    raise ValueError(f"Shape mismatch: {key}")
                target[key].copy_(value)
        model.load_state_dict(target, strict=True)
    else:
        raise ValueError("Warm start supports matching class counts or 4 -> 5")
    return sha256(path)
