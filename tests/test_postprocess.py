import numpy as np
import pytest

from pinky_lane.postprocess import correct_mask, validate_config


def scene():
    raw = np.zeros((60, 80), np.uint8)
    p = np.full((5, 60, 80), 0.025, np.float32)
    p[0] = 0.9
    return raw, p


def test_small_uncertain_opposite_lane_is_reassigned_not_deleted():
    raw, p = scene()
    raw[10:50, 20:28] = 2
    p[:, 10:50, 20:28] = 0.025
    p[2, 10:50, 20:28] = 0.9
    raw[25:28, 22:25] = 1
    p[:, 25:28, 22:25] = 0.03
    p[1, 25:28, 22:25] = 0.48
    p[2, 25:28, 22:25] = 0.43
    result, stats = correct_mask(raw, p, {})
    assert np.all(result[25:28, 22:25] == 2)
    assert stats['lane_left_to_right_pixels'] == 9
    assert np.all(result[raw == 0] == 0)


def test_high_confidence_opposite_lane_and_separate_fragments_are_preserved():
    raw, p = scene()
    raw[10:50, 20:28] = 2
    raw[25:28, 22:25] = 1
    p[1, 25:28, 22:25] = 0.9
    raw[10:12, 60:62] = 1
    result, stats = correct_mask(raw, p, {})
    np.testing.assert_array_equal(result, raw)
    assert stats['lane_left_to_right_pixels'] == 0


def test_low_score_and_small_bumps_removed_but_large_bump_preserved():
    raw, p = scene()
    raw[10:20, 10:20] = 4
    p[0, 10:20, 10:20] = 0.35
    p[4, 10:20, 10:20] = 0.5
    raw[30:32, 10:12] = 4
    p[0, 30:32, 10:12] = 0.025
    p[4, 30:32, 10:12] = 0.9
    raw[30:40, 40:50] = 4
    p[0, 30:40, 40:50] = 0.025
    p[4, 30:40, 40:50] = 0.9
    raw[:5] = 255
    result, stats = correct_mask(raw, p, {})
    assert stats['bump_rejected_pixels'] == 104
    assert np.all(result[10:20, 10:20] == 0)
    assert np.all(result[30:40, 40:50] == 4)
    assert np.all(result[:5] == 255)


def test_unknown_and_invalid_settings_are_rejected():
    for config in ({'foo': 1}, {'bump_probability': 1.1}, {'lane_neighborhood_radius': 0}):
        with pytest.raises(ValueError):
            validate_config(config)


def test_ambiguous_junction_support_does_not_flip_lane():
    raw, p = scene()
    raw[10:50, 20:28] = 2
    raw[20:30, 27:45] = 1
    raw[25:28, 22:25] = 1
    p[1] = 0.48
    p[2] = 0.43
    result, _ = correct_mask(raw, p, {})
    np.testing.assert_array_equal(result, raw)


def test_default_correction_supports_four_classes_and_preserves_inputs():
    raw, p = scene()
    original = raw.copy()
    result, stats = correct_mask(raw, p[:4], {})
    np.testing.assert_array_equal(result, original)
    np.testing.assert_array_equal(raw, original)
    assert stats['bump_rejected_pixels'] == 0
