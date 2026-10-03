"""Image-only lighting augmentation; preserve label geometry and class roles."""

import random

import numpy as np


def validate(config):
    if config.get("horizontal_flip_probability", 0) != 0:
        raise ValueError("Horizontal flips are disabled: intersection roles are ambiguous")
    if config.get("rotation_probability", 0) != 0 or config.get("rotation_degrees", 0) != 0:
        raise ValueError("This release supports image-only lighting augmentation")
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
