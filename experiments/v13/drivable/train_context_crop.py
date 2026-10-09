"""Context drivable head on a frozen lane U-Net: fuses bottleneck/d3/d2 (wide context) with d1.
Lane outputs stay exactly the parent's (same background split as LaneWithDrivable).
    python train_context.py --code DIR --data DIR --parent TS --out DIR [--neg-weight 1]"""
import argparse, json, sys, time, hashlib, random, copy
from pathlib import Path
import numpy as np, torch
from torch import nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
for k in ("code", "data", "parent", "out"): ap.add_argument("--" + k, required=True)
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--seed", type=int, default=554)
ap.add_argument("--neg-weight", type=float, default=1.0)
ap.add_argument("--head", choices=("half", "slim32", "slim16"), default="half")
ap.add_argument("--crop", type=int, default=0, help="drop this many top rows from input and mask")
a = ap.parse_args()
sys.path.insert(0, a.code)
import drivable_head as dh
from rosy_lane_model import RosyLaneDataset
out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
dev = "cuda"

def cbr(i, o, k=3, d=1):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=d * (k // 2), dilation=d, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

class ContextHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.pb, self.p3, self.p2 = cbr(256, 32, 1), cbr(64, 16, 1), cbr(32, 16, 1)
        self.fuse = nn.Sequential(cbr(64, 32), cbr(32, 32, d=2), cbr(32, 32, d=4))   # at 1/2 resolution
        self.out = nn.Conv2d(32 + 16, 1, 1)                                         # full resolution
    def forward(self, b, d3, d2, d1):
        h2 = d2.shape[-2:]
        z = torch.cat([F.interpolate(self.pb(b), size=h2, mode="bilinear", align_corners=False),
                       F.interpolate(self.p3(d3), size=h2, mode="bilinear", align_corners=False), self.p2(d2)], 1)
        z = F.interpolate(self.fuse(z), size=d1.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(torch.cat([z, d1], 1))

class SlimHead(nn.Module):
    """Context fused at 1/4 resolution; only the 1-channel logit is upsampled, plus a 1x1 on d1."""
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
    def __init__(self, lane, ignore_top, head="half"):
        super().__init__(lane, ignore_top=ignore_top)
        self.kind = head
        self.drivable = ContextHead() if head == "half" else SlimHead(32 if head == "slim32" else 16)
    def drivable_logit(self, x):
        L = self.lane
        with torch.no_grad():
            e1 = L.enc1(x); e2 = L.enc2(L.pool(e1)); e3 = L.enc3(L.pool(e2)); e4 = L.enc4(L.pool(e3))
            b = L.bottleneck(L.pool(e4))
            d4 = L.dec4(torch.cat([L.up4(b), e4], 1)); d3 = L.dec3(torch.cat([L.up3(d4), e3], 1))
            d2 = L.dec2(torch.cat([L.up2(d3), e2], 1)); d1 = L.dec1(torch.cat([L.up1(d2), e1], 1))
            lane = L.head(d1)
        if self.kind == "half":
            return lane, self.drivable(b, d3, d2, d1)
        return lane, self.drivable(b, d4, d3, d1)

class Split(RosyLaneDataset):
    def __init__(self, root, split):
        super().__init__(root, "train"); self.frames = [f for f in self.manifest["frames"] if f["split"] == split]
    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        return (x[:, a.crop:], y[a.crop:]) if a.crop else (x, y)
train, val, test = Split(a.data, "train"), Split(a.data, "val"), Split(a.data, "test")
lane, _ = dh.load_frozen_lane(a.parent, classes=train.classes[:-1])
TOP = max(0, 110 - a.crop)
model = LaneWithContext(lane, TOP, a.head).to(dev)
logf = open(out / "train.log", "w")
def p(m): print(m, flush=True); logf.write(m + "\n"); logf.flush()
nparam = sum(q.numel() for q in model.drivable.parameters())
p(f"parent {a.parent}  head params {nparam}  neg_weight {a.neg_weight}")

@torch.no_grad()
def evaluate(ds):
    model.eval(); ch = model.lane.head.out_channels; I = U = tp = fp = fn = 0; nf = [0, 0]
    for x, y in torch.utils.data.DataLoader(ds, batch_size=16):
        x, y = x.to(dev), y.to(dev); keep = y != 255; keep[..., :TOP, :] = False
        pred = model(x).argmax(1) == ch; truth = y == 5
        I += int((pred & truth & keep).sum()); U += int(((pred | truth) & keep).sum())
        tp += int((pred & truth & keep).sum()); fp += int((pred & ~truth & keep).sum()); fn += int((~pred & truth & keep).sum())
        neg = (y == 0) & keep; nf[0] += int((pred & neg).sum()); nf[1] += int(neg.sum())
    return {"iou": I / U, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "not_drivable_floor_painted": nf[0] / max(nf[1], 1)}

opt = torch.optim.Adam(model.drivable.parameters(), lr=a.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
dl = torch.utils.data.DataLoader(train, batch_size=a.batch, shuffle=True, num_workers=4)
best, best_state, hist = None, None, []; t0 = time.time()
for ep in range(1, a.epochs + 1):
    model.train(); tot = n = 0
    for x, y in dl:
        x, y = x.to(dev), y.to(dev)
        target, keep = dh.drivable_target(y, 5, 255, TOP)
        _, s = model.drivable_logit(x)
        w = torch.where(target > 0.5, 1.0, a.neg_weight)
        loss = (F.binary_cross_entropy_with_logits(s[:, 0], target, reduction="none") * w)[keep].mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(x); n += len(x)
    sched.step()
    v = evaluate(val); score = v["iou"] - 0.0 * v["not_drivable_floor_painted"]
    hist.append({"epoch": ep, "loss": tot / n, **v})
    p(f"epoch {ep}/{a.epochs} loss {tot/n:.4f} val IoU {v['iou']:.3f} P {v['precision']:.3f} R {v['recall']:.3f} not-drivable painted {v['not_drivable_floor_painted']:.3f}")
    if best is None or score > best[0]: best, best_state = (score, ep), copy.deepcopy(model.drivable.state_dict())
model.drivable.load_state_dict(best_state)
m = {"val": evaluate(val), "test": evaluate(test)}
p(f"best epoch {best[1]}  eval {json.dumps(m)}  {time.time()-t0:.0f}s")
model.eval().cpu(); x = torch.rand(1, 3, 240 - a.crop, 320)
torch.onnx.export(model, x, str(out / "model.onnx"), input_names=["input"], output_names=["logits"], opset_version=17, dynamo=False)
import onnxruntime as ort
o = ort.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"]).run(None, {"input": x.numpy()})[0]
with torch.no_grad(): ref = model(x).numpy(); lane_ref = model.lane(x).numpy()
parity = float(np.abs(o - ref).max())

torch.save({"head": model.drivable.state_dict()}, out / "context_head.pt")
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
(out / "run.json").write_text(json.dumps({"parent": a.parent, "parent_sha256": sha(a.parent), "dataset_manifest_sha256": sha(Path(a.data) / "manifest.json"),
    "head": a.head, "head_params": nparam, "train": vars(a), "best_epoch": best[1], "eval": m, "history": hist,
    "onnx_sha256": sha(out / "model.onnx"), "onnx_parity_max_abs": parity}, indent=1))
p(f"onnx parity {parity:.2e}")
