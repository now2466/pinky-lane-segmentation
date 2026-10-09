"""Assemble U-Net v13 crop128 + slim16 drivable head -> ONNX/TorchScript, parity checks, result images, manifest."""
import sys, json, hashlib, os
from pathlib import Path
import numpy as np, cv2, torch
from torch import nn
import torch.nn.functional as F
D = Path(sys.argv[1]); sys.path.insert(0, str(D / "code/training"))
import drivable_head as dh
torch.manual_seed(0)
def cbr(i, o, k=3, d=1):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=d * (k // 2), dilation=d, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
class SlimHead(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.pb, self.p4, self.p3 = cbr(256, w, 1), cbr(128, 16, 1), cbr(64, 16, 1)
        self.fuse = nn.Sequential(cbr(w + 32, w), cbr(w, w, d=2), cbr(w, w, d=4), nn.Conv2d(w, 1, 1))
        self.fine = nn.Conv2d(16, 1, 1)
    def forward(self, b, d4, d3, d1):
        h = d3.shape[-2:]
        z = torch.cat([F.interpolate(self.pb(b), size=h, mode="bilinear", align_corners=False),
                       F.interpolate(self.p4(d4), size=h, mode="bilinear", align_corners=False), self.p3(d3)], 1)
        return F.interpolate(self.fuse(z), size=d1.shape[-2:], mode="bilinear", align_corners=False) + self.fine(d1)
class LaneWithContext(dh.LaneWithDrivable):
    def __init__(self, lane):
        super().__init__(lane, ignore_top=0); self.drivable = SlimHead(16)
    def drivable_logit(self, x):
        L = self.lane
        e1 = L.enc1(x); e2 = L.enc2(L.pool(e1)); e3 = L.enc3(L.pool(e2)); e4 = L.enc4(L.pool(e3)); b = L.bottleneck(L.pool(e4))
        d4 = L.dec4(torch.cat([L.up4(b), e4], 1)); d3 = L.dec3(torch.cat([L.up3(d4), e3], 1))
        d2 = L.dec2(torch.cat([L.up2(d3), e2], 1)); d1 = L.dec1(torch.cat([L.up1(d2), e1], 1))
        return L.head(d1), self.drivable(b, d4, d3, d1)
parent_ts = D / "parent/unet_v13_crop128.torchscript.pt"
lane, n = dh.load_frozen_lane(str(parent_ts))
model = LaneWithContext(lane).eval()
model.drivable.load_state_dict(torch.load(D / "head/context_head.pt", map_location="cpu")["head"])
x = torch.rand(1, 3, 128, 320)
# exports
with torch.no_grad():
    torch.onnx.export(model, x, str(D / "model/model.onnx"), input_names=["input"], output_names=["logits"], opset_version=17, dynamo=False)
    torch.jit.trace(model, x).save(str(D / "model/lane_with_drivable_crop128.torchscript.pt"))
    torch.onnx.export(lane, x, str(D / "parent/unet_v13_crop128.onnx"), input_names=["input"], output_names=["logits"], opset_version=17, dynamo=False)
import onnxruntime as ort
sess = ort.InferenceSession(str(D / "model/model.onnx"), providers=["CPUExecutionProvider"])
psess = ort.InferenceSession(str(D / "parent/unet_v13_crop128.onnx"), providers=["CPUExecutionProvider"])
# parity on real frames
man = json.loads((D / "dataset/manifest.json").read_text())
test = [f for f in man["frames"] if f["split"] == "test"]
def load(f):
    bgr = cv2.imread(str(D / "dataset" / f["image"])); m = cv2.imread(str(D / "dataset" / f["mask"]), cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return bgr, m, rgb[112:].transpose(2, 0, 1)[None].copy()
maxdiff, lane_mismatch = 0.0, 0
for f in test[:20]:
    _, _, inp = load(f)
    o = sess.run(None, {"input": inp})[0]
    with torch.no_grad(): t = model(torch.from_numpy(inp)).numpy()
    maxdiff = max(maxdiff, float(np.abs(o - t).max()))
    po = psess.run(None, {"input": inp})[0]
    a, b = o.argmax(1), po.argmax(1)
    lane_px = b != 0
    lane_mismatch += int((a[lane_px] != b[lane_px]).sum())
print("onnx vs torch max abs", maxdiff, "lane pixels changed vs parent", lane_mismatch)
# result images
C = {0: (30, 30, 30), 1: (0, 200, 255), 2: (255, 150, 0), 3: (255, 0, 255), 4: (255, 230, 0), 5: (0, 210, 0), 255: (255, 105, 180)}
def ov(rgb, mk, top=112):
    o = rgb.astype(np.float32).copy()
    for k, c in C.items():
        s = mk == k; al = {0: 0.55, 255: 0.0}.get(k, 0.5); o[s] = o[s] * (1 - al) + np.array(c) * al
    o[:top] *= 0.45
    return o.astype(np.uint8)
(D / "results/images").mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(10); pick = [test[i] for i in sorted(rng.choice(len(test), 12, replace=False))]
val = [f for f in man["frames"] if f["split"] == "val"]; pick += [val[i] for i in sorted(rng.choice(len(val), 4, replace=False))]
tiles = []
for f in pick:
    bgr, m, inp = load(f)
    pred = np.zeros((240, 320), np.uint8); pred[112:] = sess.run(None, {"input": inp})[0].argmax(1)[0]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = np.concatenate([rgb, ov(rgb, pred), ov(rgb, np.where(np.arange(240)[:, None] < 112, 255, m))], 1)
    t = cv2.resize(t, (t.shape[1] * 2, t.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
    t = cv2.copyMakeBorder(t, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    name = Path(f["image"]).stem
    cv2.putText(t, f"{f['split']} {name}   |   original   |   prediction (U-Net v13 crop128 + drivable)   |   human label", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(D / f"results/images/{f['split']}-{name}.png"), cv2.cvtColor(t, cv2.COLOR_RGB2BGR)); tiles.append(t)
sheet = np.concatenate(tiles, 0); cv2.imwrite(str(D / "results/contact_sheet.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
# manifest
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
onnx_sha = sha(D / "model/model.onnx"); rev = f"v13-drivable-20261010-{onnx_sha[:8]}"
run = json.loads((D / "head/run.json").read_text())
classes = man["classes"]
mm = {"schema": "rosy.perception.model/1", "model_revision": rev, "task": "lane_seg",
      "files": [{"name": "model.onnx", "sha256": onnx_sha, "precision": "fp32"}],
      "input": {"shape": [1, 3, 128, 320], "layout": "nchw", "color": "rgb", "scale": 1 / 255, "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0],
                "crop": {"from_frame": [240, 320], "rows": [112, 240], "note": "resize the camera frame to 320x240 RGB, then feed rows 112..239; output row r maps to frame row 112+r; frame rows 0..111 are not predicted (treat as unknown / background)"}},
      "output": {"layout": "nchw_logits", "classes": classes},
      "dataset": {"repo": "v13-drivable-human-20261009", "revision": sha(D / "dataset/manifest.json"), "annotation_origin": "human_reviewed",
                  "note": "drivable = floor connected to the robot without crossing a white line (lane_rule_v5), every frame corrected by a human"},
      "metrics": {"test": run["eval"]["test"], "val": run["eval"]["val"]},
      "trainer": "frozen-lane-unet + slim context drivable head (1/4-res fuse of bottleneck/d4/d3, 1x1 on d1)",
      "parent_lane_model": {"model_revision": "unet-v13-crop128-20261010", "onnx_sha256": sha(D / "parent/unet_v13_crop128.onnx"),
                            "torchscript_sha256": sha(parent_ts), "checkpoint_sha256": sha(D / "parent/unet_v13_crop128_best.pt")},
      "camera_provenance": "provisional"}
(D / "model/model_manifest.json").write_text(json.dumps(mm, indent=2, ensure_ascii=False))
files = sorted(p for p in D.rglob("*") if p.is_file() and "dataset" not in p.parts and p.name != "SHA256SUMS")
(D / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(D)}\n" for p in files))
print(json.dumps({"revision": rev, "onnx_sha256": onnx_sha, "parity": maxdiff, "lane_mismatch": lane_mismatch}, indent=1))
