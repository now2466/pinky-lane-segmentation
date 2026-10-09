"""Lighting and paired positive-bump crops preserving semantic class roles."""

import random

import cv2
import numpy as np


def validate(config):
    if config.get("horizontal_flip_probability", 0) != 0:
        raise ValueError("Horizontal flips are disabled: intersection roles are ambiguous")
    if config.get("rotation_probability", 0) != 0 or config.get("rotation_degrees", 0) != 0:
        raise ValueError("Rotations are disabled: semantic lane roles must be preserved")
    crop_probability = config.get("bump_crop_probability", 0)
    crop_min = config.get("bump_crop_min_scale", .7)
    if not 0 <= crop_probability <= 1 or not .5 <= crop_min <= 1:
        raise ValueError("Invalid bump crop augmentation")
    shrink_probability = config.get("bump_shrink_probability", 0)
    shrink_low, shrink_high = config.get("bump_shrink_scale_range", [.35, .7])
    shrink_max = config.get("bump_shrink_max_pixels", 2048)
    if (not 0 <= shrink_probability <= 1 or not .1 <= shrink_low <= shrink_high <= 1
            or isinstance(shrink_max, bool) or not isinstance(shrink_max, int) or shrink_max < 1):
        raise ValueError("Invalid positive bump shrink augmentation")
    low, high = config.get("contrast_range", [0.8, 1.2])
    delta = config.get("brightness_delta", 0.06)
    probability = config.get("lighting_probability", 1.0)
    if (not all(np.isfinite(v) for v in (low, high, delta, probability))
            or not 0 < low <= high or delta < 0 or not 0 <= probability <= 1):
        raise ValueError("Invalid lighting augmentation configuration")


def augment_image(rgb, config):
    if random.random() >= config.get("lighting_probability", 1.0):
        return rgb
    low, high = config.get("contrast_range", [0.8, 1.2])
    delta = config.get("brightness_delta", 0.06)
    return np.clip(rgb * random.uniform(low, high) + random.uniform(-delta, delta), 0, 1)


def augment_bump_crop(rgb, labels, config):
    """Zoom an existing labeled positive frame; image and labels share geometry.

    This resamples the existing image and adds no native detail. Crop bounds
    contain every annotated bump pixel; nearest labels retain 255/254 sentinels.
    Semantic left/right IDs remain unchanged, with no flips or rotation.
    """
    probability = config.get("bump_crop_probability", 0)
    ys, xs = np.where(labels == 4)
    if probability == 0 or not len(xs) or random.random() >= probability:
        return rgb, labels
    height, width = labels.shape
    scale = random.uniform(config.get("bump_crop_min_scale", .7), 1)
    crop_w = max(int(width * scale), int(xs.max() - xs.min() + 1))
    crop_h = max(int(height * scale), int(ys.max() - ys.min() + 1))
    # Uniform valid origin, never omit a labeled bump at the crop boundary.
    xlo, xhi = max(0, int(xs.max()) - crop_w + 1), min(int(xs.min()), width - crop_w)
    ylo, yhi = max(0, int(ys.max()) - crop_h + 1), min(int(ys.min()), height - crop_h)
    x, y = random.randint(xlo, xhi), random.randint(ylo, yhi)
    cropped_rgb = cv2.resize(rgb[y:y + crop_h, x:x + crop_w], (width, height), interpolation=cv2.INTER_LINEAR)
    cropped_labels = cv2.resize(labels[y:y + crop_h, x:x + crop_w], (width, height), interpolation=cv2.INTER_NEAREST)
    return cropped_rgb, cropped_labels


def augment_bump_shrink(rgb, labels, config):
    """Reduce annotated positive object scale; reflected padding is unlabeled.

    This is train-only resampling of reviewed labels, not a new native annotation.
    Avoid flips/role changes and fall back if nearest sampling loses all bumps.
    The zero-probability default does not consume randomness.
    """
    probability = config.get("bump_shrink_probability", 0)
    area = int(np.count_nonzero(labels == 4))
    if probability == 0 or not 0 < area <= config.get("bump_shrink_max_pixels", 2048):
        return rgb, labels
    if random.random() >= probability:
        return rgb, labels
    low, high = config.get("bump_shrink_scale_range", [.35, .7])
    scale = random.uniform(low, high)
    height, width = labels.shape
    new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
    shrunk_labels = cv2.resize(labels, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    if not np.any(shrunk_labels == 4):
        return rgb, labels
    shrunk_rgb = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
    top, left = (height - new_height) // 2, (width - new_width) // 2
    bottom, right = height - new_height - top, width - new_width - left
    image = cv2.copyMakeBorder(shrunk_rgb, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    target = np.full(labels.shape, 255, dtype=np.uint8)
    target[top:top + new_height, left:left + new_width] = shrunk_labels
    return image, target
