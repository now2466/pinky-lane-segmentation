import numpy as np

from pinky_lane.compare_models import bump_counts
from pinky_lane.dataset import FULL_POLICY, LEGACY_POLICY


def test_unknown_legacy_background_is_not_bump_fp():
    labels = np.array([[0, 1, 255]], dtype=np.uint8)
    prediction = np.array([[4, 4, 4]], dtype=np.uint8)
    assert bump_counts(prediction, labels, LEGACY_POLICY) == {
        'tp': 0, 'fp': 1, 'fn': 0, 'iou': 0, 'recall': None}
    assert bump_counts(prediction, labels, FULL_POLICY)['fp'] == 2


def test_missing_tiny_bump_is_false_negative():
    labels = np.array([[4, 4, 0]], dtype=np.uint8)
    prediction = np.array([[4, 0, 4]], dtype=np.uint8)
    assert bump_counts(prediction, labels, FULL_POLICY) == {
        'tp': 1, 'fp': 1, 'fn': 1, 'iou': 1 / 3, 'recall': .5}
