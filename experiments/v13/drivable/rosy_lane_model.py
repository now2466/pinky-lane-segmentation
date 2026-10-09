"""Baseline trainer for rosy_lane_training.ipynb (D-373 decision 7).

Trainer-side only: imports torch at module level. export_cell.py stays torch-free.

- LaneUNet: same structure as the deployed 0930 model, so a new trainer starts
  from the same baseline.
- Preprocess: the one place the input transform is defined. RosyLaneDataset uses
  apply(); export_cell.export() gets manifest_kwargs(). The robot applies the same
  formula from the manifest: x = (pixel * scale - mean) / std, rgb flips BGR frames.
- RosyLaneDataset: follows frames[].image / frames[].mask only (dataset/1 manifest).
  Mask pixels equal to the manifest's ignore_index (D-379: 255, unlabelled) are
  allowed and left out of the loss and the IoU; train() takes it from the dataset.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

WIDTH, HEIGHT = 320, 240
COLORS = ("rgb", "bgr")


class ConvBlock(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class LaneUNet(nn.Module):
    """4-level U-Net: 1x3xHxW -> 1xCxHxW logits (H, W divisible by 16)."""

    def __init__(self, in_ch: int = 3, n_classes: int = 4, base: int = 16):
        super().__init__()
        c = [base * 2 ** i for i in range(5)]
        self.enc1, self.enc2 = ConvBlock(in_ch, c[0]), ConvBlock(c[0], c[1])
        self.enc3, self.enc4 = ConvBlock(c[1], c[2]), ConvBlock(c[2], c[3])
        self.bottleneck = ConvBlock(c[3], c[4])
        self.pool = nn.MaxPool2d(2)
        self.up4, self.dec4 = nn.ConvTranspose2d(c[4], c[3], 2, stride=2), ConvBlock(c[4], c[3])
        self.up3, self.dec3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2), ConvBlock(c[3], c[2])
        self.up2, self.dec2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2), ConvBlock(c[2], c[1])
        self.up1, self.dec1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2), ConvBlock(c[1], c[0])
        self.head = nn.Conv2d(c[0], n_classes, 1)

    def forward(self, x):
        return self.head(self.features(x))

    def features(self, x):
        """Last decoder features (N x base x H x W), the input of head."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        return self.dec1(torch.cat([self.up1(d2), e1], 1))


@dataclass(frozen=True)
class Preprocess:
    color: str = "rgb"
    scale: float = 1 / 255
    mean: tuple = (0.0, 0.0, 0.0)
    std: tuple = (1.0, 1.0, 1.0)

    def __post_init__(self):
        if self.color not in COLORS:
            raise ValueError(f"color must be one of {COLORS}")
        object.__setattr__(self, "scale", float(self.scale))
        object.__setattr__(self, "mean", tuple(float(v) for v in self.mean))
        object.__setattr__(self, "std", tuple(float(v) for v in self.std))
        if len(self.mean) != 3 or len(self.std) != 3 or min(self.std) <= 0 or self.scale <= 0:
            raise ValueError("mean/std need 3 values, std and scale must be > 0")

    def apply(self, bgr: np.ndarray) -> np.ndarray:
        """HxWx3 uint8 BGR (OpenCV) -> 3xHxW float32, exactly as the robot does it."""
        img = bgr[..., ::-1] if self.color == "rgb" else bgr   # same ops as lane_mask.preprocess
        x = img.astype(np.float32) * np.float32(self.scale)
        x = (x - np.asarray(self.mean, np.float32)) / np.asarray(self.std, np.float32)
        return np.ascontiguousarray(x.transpose(2, 0, 1))

    def manifest_kwargs(self) -> dict:
        return {"color": self.color, "scale": self.scale,
                "mean": list(self.mean), "std": list(self.std)}


class RosyLaneDataset(torch.utils.data.Dataset):
    """One split of a rosy.perception.dataset/1 folder. Never re-split: splits are per session."""

    def __init__(self, root, split, *, color="rgb", scale=1 / 255,
                 mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.classes = self.manifest["classes"]
        self.frames = [f for f in self.manifest["frames"] if f["split"] == split]
        # Unlabelled mask value (D-379 addendum 2026-10-01); None when the manifest has none.
        ignore = self.manifest.get("ignore_index")
        if ignore is not None and (isinstance(ignore, bool) or not isinstance(ignore, int)
                                   or not len(self.classes) <= ignore <= 255):
            raise ValueError(f"manifest ignore_index {ignore!r}: an integer in "
                             f"[{len(self.classes)}, 255], never a class index")
        self.ignore_index = ignore
        self.preprocess = Preprocess(color, scale, mean, std)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        f = self.frames[i]
        bgr = cv2.imread(str(self.root / f["image"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.root / f["mask"]), cv2.IMREAD_UNCHANGED)
        if bgr is None or mask is None:
            raise FileNotFoundError(f"unreadable frame {f['image']} / {f['mask']}")
        bad = mask.ndim != 2 or bool(np.any((mask >= len(self.classes)) & (mask != (
            -1 if self.ignore_index is None else self.ignore_index))))
        if bad:
            allowed = "" if self.ignore_index is None else f" or ignore_index {self.ignore_index}"
            raise ValueError(f"{f['mask']}: mask must be single-channel class indexes "
                             f"< {len(self.classes)}{allowed} (shape {mask.shape}, max {int(mask.max())})")
        bgr = cv2.resize(bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(self.preprocess.apply(bgr)), torch.from_numpy(mask.astype(np.int64))


def class_mismatch(classes, dataset_classes) -> str | None:
    """None when [(name, role), ...] equals the dataset classes in index order, else why not."""
    want = [(str(n), str(r)) for n, r in classes]
    have = [(c["name"], c["role"]) for c in sorted(dataset_classes, key=lambda c: c["index"])]
    if want == have:
        return None
    if sorted(want) == sorted(have):
        return f"class order differs: dataset has {have}, form has {want}"
    return f"classes differ: dataset has {have}, form has {want}"


def _iou_counts(pred, target, n, ignore_index=None):
    pred, target = pred.reshape(-1), target.reshape(-1)
    if ignore_index is not None:
        keep = target != ignore_index
        pred, target = pred[keep], target[keep]
    inter = torch.bincount(target[pred == target], minlength=n)[:n]
    area_p = torch.bincount(pred, minlength=n)[:n]
    area_t = torch.bincount(target, minlength=n)[:n]
    return inter, area_p + area_t - inter


def _ratios(inter, union) -> list:
    return [None if int(u) == 0 else float(i) / float(u) for i, u in zip(inter, union)]


def class_iou(pred, target, n, ignore_index=None) -> list:
    """Per-class IoU over all pixels. None where a class is absent from both pred and target."""
    return _ratios(*_iou_counts(pred, target, n, ignore_index))


def _mean_iou(val_iou: dict) -> float:
    vals = [v for v in val_iou.values() if v is not None]
    return sum(vals) / len(vals) if vals else -1.0


def train(model, train_ds, val_ds, *, epochs, lr, batch_size, device, ignore_index=None,
          num_workers=None, log=print, on_epoch: Callable[[dict], None] | None = None,
          loss_fn=None, optimizer_factory=None) -> dict:
    """Adam + cross-entropy; the model ends with the best epoch's weights (mean non-None val IoU).
    ignore_index defaults to the dataset's (manifest ignore_index): those pixels count in
    neither the loss nor the IoU. on_epoch (e.g. an experiment tracker) gets each history
    row plus "mean_val_iou" (None when no class has an IoU yet).
    Optional loss_fn and optimizer_factory(parameters, lr=...) select a training recipe.

    Returns {history: [{epoch, train_loss, val_loss, val_iou{name: iou}}], best_epoch, val_iou}."""
    if ignore_index is None:
        ignore_index = getattr(train_ds, "ignore_index", None)
    names = [c["name"] for c in sorted(val_ds.classes, key=lambda c: c["index"])]
    n = len(names)
    model = model.to(device)
    opt = (optimizer_factory or torch.optim.Adam)(model.parameters(), lr=lr)
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100 if ignore_index is None else ignore_index)
    if num_workers is None:
        num_workers = 0 if os.name == "nt" else 2
    loader = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=str(device).startswith("cuda"))
    train_dl = torch.utils.data.DataLoader(train_ds, shuffle=True, **loader)
    val_dl = torch.utils.data.DataLoader(val_ds, **loader)
    history, best, best_state = [], None, None
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total, count = total + loss.item() * len(x), count + len(x)
        model.eval()
        inter = torch.zeros(n, dtype=torch.long)
        union = torch.zeros(n, dtype=torch.long)
        vtotal, vcount = 0.0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                vtotal, vcount = vtotal + loss_fn(logits, y).item() * len(x), vcount + len(x)
                i, u = _iou_counts(logits.argmax(1).cpu(), y.cpu(), n, ignore_index)
                inter, union = inter + i, union + u
        row = {"epoch": epoch, "train_loss": total / max(count, 1),
               "val_loss": vtotal / max(vcount, 1),
               "val_iou": dict(zip(names, _ratios(inter, union)))}
        history.append(row)
        if on_epoch is not None:
            mean = _mean_iou(row["val_iou"])
            on_epoch({**row, "val_iou": dict(row["val_iou"]),
                      "mean_val_iou": None if mean < 0 else mean})
        if best is None or _mean_iou(row["val_iou"]) > _mean_iou(best["val_iou"]):
            best, best_state = row, copy.deepcopy(model.state_dict())
        if log:
            ious = " ".join(f"{k}={'-' if v is None else f'{v:.3f}'}" for k, v in row["val_iou"].items())
            log(f"epoch {epoch}/{epochs} loss {row['train_loss']:.4f} val_loss {row['val_loss']:.4f} {ious}")
    if best_state is not None:
        model.load_state_dict(best_state)
        if log:
            log(f"best epoch {best['epoch']} (mean val IoU {_mean_iou(best['val_iou']):.3f}) restored")
    return {"history": history, "best_epoch": best["epoch"] if best else None,
            "val_iou": best["val_iou"] if best else {}}
