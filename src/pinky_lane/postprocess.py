"""Opt-in conservative spatial corrections; no screen-side lane assignment."""
import cv2
import numpy as np

DEFAULTS = {
    "bump_probability": 0.75, "bump_margin": 0.25, "bump_min_area": 48,
    "lane_max_island_area": 32, "lane_max_parent_fraction": 0.08,
    "lane_neighborhood_radius": 3, "lane_min_support": 12,
    "lane_neighbor_agreement": 0.90, "lane_max_score_margin": 0.30,
    "lane_min_opposite_probability": 0.15,
}


def validate_config(config):
    if set(config) - set(DEFAULTS):
        raise ValueError("Unknown postprocessing setting")
    result = {**DEFAULTS, **config}
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"Invalid postprocessing value: {key}")
        if key.endswith(('area', 'radius', 'support')):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"Expected positive integer: {key}")
        elif not 0 <= value <= 1:
            raise ValueError(f"Expected value in [0,1]: {key}")
    return result


def components(binary):
    return cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)


def correct_mask(raw, probabilities, config):
    cfg = validate_config(config)
    mask = raw.copy()
    bump_rejected = np.zeros(raw.shape, bool)
    if probabilities.shape[0] == 5:
        other = probabilities[:4].max(axis=0)
        bump_rejected = (raw == 4) & ((probabilities[4] < cfg['bump_probability']) |
                                      (probabilities[4] - other < cfg['bump_margin']))
        count, labels, stats, _ = components((raw == 4) & ~bump_rejected)
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] < cfg['bump_min_area']:
                bump_rejected |= labels == index
        fallback = probabilities[:4].argmax(axis=0).astype(np.uint8)
        mask[bump_rejected] = fallback[bump_rejected]
    before_lane = mask.copy()
    _, parent_labels, parent_stats, _ = components((mask == 1) | (mask == 2))
    radius = cfg['lane_neighborhood_radius']
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    for source, target in ((1, 2), (2, 1)):
        count, labels, stats, _ = components(before_lane == source)
        for index in range(1, count):
            area = stats[index, cv2.CC_STAT_AREA]
            if area > cfg['lane_max_island_area']:
                continue
            island = labels == index
            parent_id = parent_labels[island][0]
            if not parent_id or area / parent_stats[parent_id, cv2.CC_STAT_AREA] > cfg['lane_max_parent_fraction']:
                continue
            nearby = cv2.dilate(island.astype(np.uint8), kernel).astype(bool) & ~island
            nearby &= parent_labels == parent_id
            support = np.count_nonzero(nearby & (before_lane == target))
            total = np.count_nonzero(nearby)
            if support < cfg['lane_min_support'] or support / max(total, 1) < cfg['lane_neighbor_agreement']:
                continue
            if np.any(probabilities[source][island] - probabilities[target][island] > cfg['lane_max_score_margin']):
                continue
            if np.any(probabilities[target][island] < cfg['lane_min_opposite_probability']):
                continue
            mask[island] = target
    return mask, {
        'bump_rejected_pixels': int(bump_rejected.sum()),
        'lane_left_to_right_pixels': int(((before_lane == 1) & (mask == 2)).sum()),
        'lane_right_to_left_pixels': int(((before_lane == 2) & (mask == 1)).sum()),
    }
