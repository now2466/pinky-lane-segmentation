import sys, os, json, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "unet-src"))
from pinky_lane.dataset import LaneDataset
from pinky_lane.checkpoints import load_model
CROP = 112; dev = "cuda"
BANDS = {"rows 112-143 (far)": (112, 144), "rows 144-239 (near, steering 40%)": (144, 240), "rows 112-239 (all)": (112, 240)}
src = LaneDataset("data-v13", "test", 5)
res = {}
for name, path, crop in [("original", "runs/unet/best.pt", False), ("crop128", "runs/unet-crop128/best.pt", True)]:
    model, _, _, _ = load_model(path, dev, ignore_top=0); model.eval()
    inter = {b: np.zeros(5) for b in BANDS}; union = {b: np.zeros(5) for b in BANDS}
    with torch.no_grad():
        for i in range(len(src)):
            im, lab, pol, _ = src[i]
            x = (im[:, CROP:] if crop else im)[None].to(dev)
            p = model(x).argmax(1)[0].cpu().numpy()
            if crop: p = np.concatenate([np.zeros((CROP, 320), np.int64), p], 0)
            l = lab.numpy()
            nc = 4 if pol == "legacy_partial_class4" else 5
            for b, (y0, y1) in BANDS.items():
                pp, ll = p[y0:y1], l[y0:y1]; v = ll != 255
                for c in range(1, nc):
                    inter[b][c] += ((pp == c) & (ll == c) & v).sum(); union[b][c] += (((pp == c) | (ll == c)) & v).sum()
    res[name] = {b: {n: round(float(inter[b][c] / union[b][c]), 4) if union[b][c] else None
                     for c, n in enumerate(["bg", "lane_left", "lane_right", "crosswalk", "speed_bump"]) if c} for b in BANDS}
print(json.dumps(res, indent=1))
