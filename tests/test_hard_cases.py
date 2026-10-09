import numpy as np

from pinky_lane.dataset import FULL_POLICY, LEGACY_POLICY
from pinky_lane.hard_cases import error_counts, run
from .helpers import synthetic_dataset, make_checkpoint


def test_error_mining_respects_unknown_and_ignored_pixels():
    labels = np.array([[0, 254, 1, 2, 255]], np.uint8)
    pred = np.array([[4, 4, 2, 1, 4]], np.uint8)
    legacy = error_counts(pred, labels, LEGACY_POLICY)
    assert legacy['speed_bump_false_positive_pixels'] == 1
    assert legacy['lane_role_confusion_pixels'] == 2
    assert legacy['foreground_error_pixels'] == 3
    full = error_counts(pred, labels, FULL_POLICY)
    assert full['speed_bump_false_positive_pixels'] == 2
    assert full['foreground_error_pixels'] == 4


def test_hard_case_pipeline_emits_measured_frames_and_no_dataset_changes(tmp_path):
    root = synthetic_dataset(tmp_path / 'data', 5)
    checkpoint = tmp_path / 'model.pt'
    make_checkpoint(checkpoint, 5, {'ignore_top': 0})
    original = (root / 'manifest.json').read_bytes()
    report = run(checkpoint, root, 'val', tmp_path / 'cases', 'cpu', 2)
    assert report['evaluated_frames'] == 1
    assert len(report['frames']) == 1
    assert (root / 'manifest.json').read_bytes() == original
    assert (tmp_path / 'cases/report.json').exists()
