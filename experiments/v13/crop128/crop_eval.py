"""Test-split metrics for the cropped model vs the original full-frame model, both scored on rows 112-239."""
import sys, os, json, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "unet-src"))
from torch.utils.data import DataLoader
from pinky_lane.dataset import LaneDataset
from pinky_lane.metrics import evaluate
from pinky_lane.checkpoints import load_model
CROP = 112
dev = torch.device("cuda")
class Crop(torch.utils.data.Dataset):
    def __init__(s, d, crop_input): s.d, s.c = d, crop_input
    def __len__(s): return len(s.d)
    def __getitem__(s, i):
        im, l, pol, g = s.d[i]; l = l.clone(); l[:CROP] = 255
        if s.c: im, l = im[:, CROP:], l[CROP:]
        return im, l, pol, g
src = LaneDataset("data-v13", "test", 5)
out = {}
for name, path, crop in [("original_full_frame", "runs/unet/best.pt", False), ("crop128", "runs/unet-crop128/best.pt", True)]:
    model, _, _, _ = load_model(path, "cuda", ignore_top=0)
    r = evaluate(model, DataLoader(Crop(src, crop), batch_size=8), dev, 5)["fixed_foreground"]
    out[name] = {"mean_iou": r["foreground_mean_iou"], **{k: v["iou"] for k, v in r["classes"].items()},
                 "lane_role_confusion_rate": r.get("lane_role_confusion_rate")}
print(json.dumps(out, indent=1))
json.dump(out, open("results/crop128-vs-full-test.json", "w"), indent=1)
