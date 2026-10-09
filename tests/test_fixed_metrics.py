import pytest
import torch

from pinky_lane.dataset import FULL_POLICY, LEGACY_POLICY
from pinky_lane.metrics import fixed_foreground_metrics, selection_score


def test_absent_class_false_positive_does_not_change_selected_class_count():
    clean = torch.diag(torch.tensor([100, 10, 10, 0, 10]))
    noisy = clean.clone()
    noisy[0, 0] -= 1
    noisy[0, 3] += 1
    first = fixed_foreground_metrics({FULL_POLICY: clean}, 5)
    second = fixed_foreground_metrics({FULL_POLICY: noisy}, 5)
    assert first['included_classes'] == second['included_classes'] == ['lane_left', 'lane_right', 'speed_bump']
    assert first['foreground_mean_iou'] == second['foreground_mean_iou'] == 1
    assert second['classes']['crosswalk']['false_positive_pixels'] == 1
    assert second['classes']['crosswalk']['false_positive_rate'] == pytest.approx(1 / 130)
    assert second['classes']['crosswalk']['precision'] == 0


def test_legacy_lanes_and_crosswalk_included_but_bumps_not_evaluated():
    full = torch.diag(torch.tensor([100, 10, 10, 0, 10]))
    legacy = torch.diag(torch.tensor([100, 10, 10, 10]))
    legacy[2, 2] -= 4
    legacy[2, 1] += 4
    result = fixed_foreground_metrics({FULL_POLICY: full, LEGACY_POLICY: legacy}, 5)
    assert result['included_classes'] == ['lane_left', 'lane_right', 'crosswalk', 'speed_bump']
    assert result['classes']['speed_bump']['valid_pixels'] == 130
    assert result['classes']['crosswalk']['gt_pixels'] == 10
    assert result['classes']['lane_left']['iou'] == pytest.approx(20/24)
    assert result['classes']['lane_right']['recall'] == .8
    assert result['lane_role_confusion_pixels'] == 4
    assert result['lane_role_confusion_rate'] == .1
    assert selection_score({'fixed_foreground': result}, 5, 'fixed_foreground') == result['foreground_mean_iou']


def test_unknown_selection_metric_rejected():
    with pytest.raises(ValueError, match='Unknown'):
        selection_score({}, 5, 'typo')
