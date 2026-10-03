"""Shared RGB preprocessing and overlays that preserve the original background."""

import cv2
import numpy as np
import torch

COLORS_BGR = {1: (255, 255, 0), 2: (0, 140, 255), 3: (255, 0, 255), 4: (0, 255, 0)}


def preprocess(bgr):
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected a color image")
    small = cv2.resize(bgr, (320, 240), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))


@torch.inference_mode()
def predict(model, frames, device, ignore_top):
    batch = torch.stack([preprocess(frame) for frame in frames]).to(device)
    masks = model(batch).argmax(1).cpu().numpy().astype(np.uint8)
    masks[:, :ignore_top] = 255
    return masks


def overlay(bgr, mask, alpha=0.55):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0,1]")
    large = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    result = bgr.copy()
    for class_id, color in COLORS_BGR.items():
        selected = large == class_id
        result[selected] = (bgr[selected].astype(np.float32) * (1-alpha)
                            + np.asarray(color) * alpha).astype(np.uint8)
    return result


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to save {path}")
