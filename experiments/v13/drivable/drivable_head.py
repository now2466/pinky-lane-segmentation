"""Frozen lane model + trained drivable head (user decision 2026-10-07).

A delivered lane model (now2466 pinky-lane-segmentation v11, TorchScript) stays
frozen; only a small head on its last decoder features learns drivable. The
exported model keeps every lane class's logit as delivered and appends one
drivable channel, so argmax gives the lane model's own answer everywhere it does
not say background, and splits its background into background / drivable:

    background' = bg - relu(s)    drivable = bg - relu(-s)    (s = head logit)

max(background', drivable) == bg, so lane pixels never change (an exact tie
between bg and a lane class with s > 0 goes to the lane class instead of drivable). Rows above
ignore_top are forced to background, as the delivered v11 ROI wrapper does.

Direct CLI export is held until this head is called inside the trusted D-464
IndexedReview training admission. The dataset needs exactly one class with role "drivable"; its other classes only
count as "not drivable", and its ignore_index pixels are left out. Val IoU is
scored where the lane model says background (the only pixels the head can
change); the IoU over all labelled pixels is reported beside it, and the outside-band FP:
the fraction of label-0 pixels in rows that hold drivable labels (D-554 item 9 band, not
walls) that the model calls drivable.
"""

from __future__ import annotations

import copy

if __name__ == "__main__":
    raise SystemExit("v13-drivable training requires trusted owner IndexedReview admission")

import torch
from torch import nn

from rosy_lane_model import HEIGHT, WIDTH, LaneUNet

# State-dict names of the pinky-lane-segmentation LaneUNet -> rosy_lane_model.LaneUNet.
KEY_RENAMES = ((".body.", "."), ("middle.", "bottleneck."))
PARITY_MAX_ABS = 1e-5


def load_frozen_lane(path, *, classes=None) -> tuple[LaneUNet, int]:
    """TorchScript lane model -> frozen LaneUNet with the same output (checked), and its class count."""
    script = torch.jit.load(str(path), map_location="cpu").eval()
    state = {}
    for key, value in script.state_dict().items():
        for old, new in KEY_RENAMES:
            key = key.replace(old, new)
        state[key] = value
    n_classes, base = state["head.weight"].shape[0], state["enc1.0.weight"].shape[0]
    lane = LaneUNet(n_classes=n_classes, base=base)
    lane.load_state_dict(state, strict=True)
    lane.eval().requires_grad_(False)
    x = torch.rand(1, 3, HEIGHT, WIDTH, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        diff = float((lane(x) - script(x)).abs().max())
    if diff > PARITY_MAX_ABS:
        raise ValueError(f"{path}: rebuilt lane model differs from the TorchScript by {diff}")
    if classes is not None and len(classes) != n_classes:
        raise ValueError(f"{path}: {n_classes} output channels, {len(classes)} classes given")
    return lane, n_classes


def verify_parent_parity(lane, torchscript_path, onnx_path, frames, *, ignore_top=0):
    """Check frozen lane and delivered ONNX on admitted camera inputs before head training."""
    import numpy as np
    import onnxruntime as ort

    script = torch.jit.load(str(torchscript_path), map_location="cpu").eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    maximum = 0.0
    ambiguous = 0
    with torch.no_grad():
        for x in frames:
            x = x.cpu()
            reference = script(x).numpy()
            rebuilt = lane.eval()(x).numpy()
            delivered = session.run(None, {input_name: x.numpy()})[0]
            if (reference.shape != rebuilt.shape or reference.shape != delivered.shape
                    or not all(np.isfinite(v).all() for v in (reference, rebuilt, delivered))):
                raise ValueError("parent TorchScript/ONNX shape or finite logits differ")
            # The delivered v11 ONNX carries the ROI wrapper (rows above ignore_top forced to
            # background), so only rows at and below ignore_top are compared with it.
            maximum = max(maximum, float(np.max(np.abs(reference - rebuilt))),
                          float(np.max(np.abs(reference[:, :, ignore_top:] - delivered[:, :, ignore_top:]))))
            mismatch = reference.argmax(1)[:, ignore_top:] != delivered.argmax(1)[:, ignore_top:]
            ordered = np.sort(reference[:, :, ignore_top:], axis=1)
            margin = ordered[:, -1] - ordered[:, -2]
            error = np.max(np.abs(reference[:, :, ignore_top:] - delivered[:, :, ignore_top:]), axis=1)
            unstable = mismatch & (margin <= 2 * error + 1e-7)
            ambiguous += int(unstable.sum())
            if maximum > 1e-3 or np.any(mismatch & ~unstable):
                raise ValueError(f"parent TorchScript/ONNX logits or lane pixels differ: max_abs={maximum}")
    return {"samples": len(frames), "max_abs": maximum, "ambiguous_pixels": ambiguous}


def verify_candidate_lane_parity(parent_onnx, candidate_onnx, frames, *, ignore_top=0):
    """A new drivable channel may split only parent background on admitted inputs."""
    import numpy as np
    import onnxruntime as ort

    parent = ort.InferenceSession(str(parent_onnx), providers=["CPUExecutionProvider"])
    candidate = ort.InferenceSession(str(candidate_onnx), providers=["CPUExecutionProvider"])
    checked = ambiguous = 0
    for frame in frames:
        x = frame.cpu().numpy()
        original = parent.run(None, {parent.get_inputs()[0].name: x})[0]
        output = candidate.run(None, {candidate.get_inputs()[0].name: x})[0]
        if (output.shape != (original.shape[0], original.shape[1] + 1, *original.shape[2:])
                or not np.isfinite(original).all() or not np.isfinite(output).all()):
            raise ValueError("candidate ONNX output shape or finite logits differ")
        parent_pixels = original.argmax(1)
        candidate_pixels = output.argmax(1)
        mapped = np.where(candidate_pixels == original.shape[1], 0, candidate_pixels)
        mismatch = mapped[:, ignore_top:] != parent_pixels[:, ignore_top:]
        ordered = np.sort(original[:, :, ignore_top:], axis=1)
        margin = ordered[:, -1] - ordered[:, -2]
        rebuilt = np.concatenate((np.maximum(output[:, :1], output[:, -1:]),
                                  output[:, 1:-1]), axis=1)
        error = np.max(np.abs(original[:, :, ignore_top:] - rebuilt[:, :, ignore_top:]), axis=1)
        if np.max(error) > 1e-3:
            raise ValueError("candidate ONNX changes parent lane pixels or logits")
        unstable = mismatch & (margin <= 2 * error + 1e-7)
        ambiguous += int(unstable.sum())
        if np.any(mismatch & ~unstable):
            raise ValueError("candidate ONNX changes parent lane pixels")
        if ignore_top and np.any(candidate_pixels[:, :ignore_top] != 0):
            raise ValueError("candidate ONNX paints ignored top rows")
        checked += 1
    if not checked:
        raise ValueError("candidate parity requires admitted frames")
    return {"samples": checked, "parent_lane_pixels_preserved": True,
            "ambiguous_pixels": ambiguous}


class LaneWithDrivable(nn.Module):
    """1x3xHxW -> 1x(C+1)xHxW logits: the lane model's C channels, then drivable."""

    def __init__(self, lane: LaneUNet, *, ignore_top: int = 0):
        super().__init__()
        if not 0 <= ignore_top < HEIGHT:
            raise ValueError(f"ignore_top must be in [0, {HEIGHT - 1}]")
        self.lane, self.ignore_top = lane, int(ignore_top)
        width = lane.head.in_channels
        self.drivable = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False), nn.BatchNorm2d(width),
            nn.ReLU(inplace=True), nn.Conv2d(width, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.lane.eval()  # frozen: its BatchNorm statistics never move
        return self

    def drivable_logit(self, x):
        """(lane logits, head logit s) for the same features."""
        with torch.no_grad():
            features = self.lane.features(x)
            lane = self.lane.head(features)
        return lane, self.drivable(features)

    def forward(self, x):
        lane, s = self.drivable_logit(x)
        bg = lane[:, :1]
        out = torch.cat([bg - torch.relu(s), lane[:, 1:], bg - torch.relu(-s)], 1)
        if self.ignore_top:
            top = out[:, :, :self.ignore_top]
            peak = top.amax(dim=1, keepdim=True)
            forced = torch.cat([peak + 1.0, top[:, 1:] - 1000.0], 1)
            out = torch.cat([forced, out[:, :, self.ignore_top:]], 2)
        return out


def drivable_target(mask, drivable_index: int, ignore_index, ignore_top: int = 0):
    """Class-index mask -> (float target 1/0, bool keep)."""
    keep = torch.ones_like(mask, dtype=torch.bool)
    if ignore_index is not None:
        keep &= mask != ignore_index
    if ignore_top:
        keep[..., :ignore_top, :] = False
    return (mask == drivable_index).float(), keep


def _drivable_index(dataset) -> int:
    found = [c["index"] for c in dataset.classes if c["role"] == "drivable"]
    if len(found) != 1:
        raise ValueError(f"dataset needs exactly one class with role drivable, has {len(found)}")
    return found[0]


def train_head(model: LaneWithDrivable, train_ds, val_ds, *, epochs, lr, batch_size, device,
               log=print) -> dict:
    """Adam on the head only, BCE on labelled pixels; ends with the best val drivable IoU's weights."""
    index, ignore = _drivable_index(train_ds), train_ds.ignore_index
    if _drivable_index(val_ds) != index:
        raise ValueError("train and val disagree on the drivable class index")
    model = model.to(device)
    drivable_channel = model.lane.head.out_channels
    opt = torch.optim.Adam(model.drivable.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    loader = dict(batch_size=batch_size, num_workers=0)
    train_dl = torch.utils.data.DataLoader(train_ds, shuffle=True, **loader)
    val_dl = torch.utils.data.DataLoader(val_ds, **loader)
    history, best, best_state = [], None, None
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            target, keep = drivable_target(y, index, ignore, model.ignore_top)
            if not keep.any():
                continue
            _, s = model.drivable_logit(x)
            loss = bce(s[:, 0], target)[keep].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, count = total + loss.item() * len(x), count + len(x)
        model.eval()
        counts = {"all": [0, 0], "lane_background": [0, 0]}
        band_fp = [0, 0]
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                target, keep = drivable_target(y, index, ignore, model.ignore_top)
                answer = model(x).argmax(1)
                pred, truth = answer == drivable_channel, target.bool()
                # The head only splits the lane model's background; paint or crosswalk
                # labelled drivable can never come out drivable, so score both ways.
                lane_bg = (answer == 0) | pred
                for name, scope in (("all", keep), ("lane_background", keep & lane_bg)):
                    counts[name][0] += int((pred & truth & scope).sum())
                    counts[name][1] += int(((pred | truth) & scope).sum())
                band = (y == 0) & keep & truth.any(dim=-1, keepdim=True)
                band_fp[0] += int((pred & band).sum())
                band_fp[1] += int(band.sum())
        ious = {k: (i / u if u else None) for k, (i, u) in counts.items()}
        iou = ious["lane_background"]
        row = {"epoch": epoch, "train_loss": total / max(count, 1), "val_drivable_iou": iou,
               "val_drivable_iou_all": ious["all"],
               "val_outside_band_fp": band_fp[0] / band_fp[1] if band_fp[1] else None}
        history.append(row)
        if iou is not None and (best is None or iou > best["val_drivable_iou"]):
            best, best_state = row, copy.deepcopy(model.drivable.state_dict())
        if log:
            fmt = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
            log(f"epoch {epoch}/{epochs} loss {row['train_loss']:.4f} "
                f"drivable={fmt(iou)} (all labelled pixels {fmt(ious['all'])}) "
                f"outside-band FP {fmt(row['val_outside_band_fp'])}")
    if best_state is not None:
        model.drivable.load_state_dict(best_state)
    return {"history": history, "best_epoch": best["epoch"] if best else None,
            "val_drivable_iou": best["val_drivable_iou"] if best else None,
            "val_drivable_iou_all": best["val_drivable_iou_all"] if best else None,
            "val_outside_band_fp": best["val_outside_band_fp"] if best else None}
