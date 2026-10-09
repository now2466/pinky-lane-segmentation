"""Trace the cropped U-Net (input 1x3x128x320) to TorchScript with a sidecar like pinky_lane.export."""
import sys, os, json, hashlib, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "unet-src"))
from pinky_lane.checkpoints import load_model
model, _, count, _ = load_model("runs/unet-crop128/best.pt", "cpu", ignore_top=0)
torch.manual_seed(17); x = torch.rand(1, 3, 128, 320)
with torch.inference_mode():
    ref = model(x); tr = torch.jit.trace(model, x); err = float((ref - tr(x)).abs().max())
out = "runs/unet-crop128/unet_v13_crop128.torchscript.pt"; tr.save(out)
with torch.inference_mode(): rel = float((ref - torch.jit.load(out)(x)).abs().max())
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
meta = {"classes": ["background", "lane_left", "lane_right", "crosswalk", "speed_bump"][:count], "input_shape": [1, 3, 128, 320],
        "input_color": "RGB", "input_range": [0, 1], "input_crop": {"source_size": [240, 320], "rows": [112, 240], "note": "resize to 320x240, then take rows 112..239"},
        "output": "logits", "output_shape": [1, count, 128, 320], "inference": {"ignore_top": 0},
        "checkpoint_sha256": sha("runs/unet-crop128/best.pt"), "torchscript_sha256": sha(out), "torch_version": torch.__version__,
        "trace_max_abs_error": err, "reload_max_abs_error": rel}
json.dump(meta, open(out + ".json", "w"), indent=2); print(json.dumps(meta, indent=1))
