import random

import numpy as np
import pytest
import torch

from pinky_lane.augmentations import augment_bump_crop, validate
from pinky_lane.dataset import FULL_POLICY, LEGACY_POLICY
from pinky_lane.losses import bump_positive_loss


def test_positive_loss_uses_only_labeled_positive_pixels_and_balances_frames():
    logits = torch.zeros(2, 5, 2, 2, requires_grad=True)
    labels = torch.tensor([[[4, 255], [0, 1]], [[4, 4], [4, 0]]])
    loss = bump_positive_loss(logits, labels, [FULL_POLICY] * 2, 1, 2)
    assert float(loss.detach()) == pytest.approx(float(np.log(5)) * 1.5)
    loss.backward()
    assert logits.grad[0, :, 0, 1].abs().sum() == 0
    assert logits.grad[0, :, 1, 0].abs().sum() == 0
    assert logits.grad[0, 4, 0, 0] < 0


def test_no_legacy_positive_supervision_and_zero_gradient():
    logits = torch.zeros(1, 5, 2, 2, requires_grad=True)
    labels = torch.zeros(1, 2, 2, dtype=torch.long)
    loss = bump_positive_loss(logits, labels, [LEGACY_POLICY])
    loss.backward()
    assert loss == 0
    assert torch.count_nonzero(logits.grad) == 0


def test_crop_keeps_all_bump_components_and_label_sentinels():
    random.seed(14)
    labels = np.zeros((240, 320), dtype=np.uint8)
    labels[50:52, 40:44] = 4
    labels[180:182, 260:263] = 4
    labels[210:, :] = 255
    labels[100:110, :] = 254
    rgb = np.zeros((240, 320, 3), dtype=np.float32)
    rgb[labels == 4] = 1
    for _ in range(20):
        image, target = augment_bump_crop(rgb, labels, {'bump_crop_probability': 1, 'bump_crop_min_scale': .5})
        assert target.shape == labels.shape
        assert image.shape == rgb.shape
        assert set(np.unique(target)) <= {0, 4, 254, 255}
        assert (target == 4).sum() >= (labels == 4).sum()
        assert (image[target == 4] > 0).any()


def test_crop_disabled_on_negative_frames_and_invalid_settings():
    labels = np.zeros((240, 320), dtype=np.uint8)
    image = np.zeros((240, 320, 3), dtype=np.float32)
    a, b = augment_bump_crop(image, labels, {'bump_crop_probability': 1})
    assert a is image and b is labels
    with pytest.raises(ValueError):
        validate({'bump_crop_min_scale': .1})


def test_disabled_crop_preserves_random_sequence_on_positive_frame():
    image = np.zeros((240, 320, 3), dtype=np.float32)
    labels = np.zeros((240, 320), dtype=np.uint8)
    labels[140, 140] = 4
    random.seed(27)
    expected = random.random()
    random.seed(27)
    a, b = augment_bump_crop(image, labels, {})
    assert a is image and b is labels
    assert random.random() == expected


def test_shrink_creates_smaller_positive_without_labeling_reflected_padding():
    from pinky_lane.augmentations import augment_bump_shrink
    labels = np.zeros((240, 320), dtype=np.uint8)
    labels[130:140, 100:140] = 4
    labels[180:220, 20:22] = 1
    labels[180:220, 200:202] = 2
    image = np.ones((240, 320, 3), dtype=np.float32)
    a, b = augment_bump_shrink(image, labels, {
        'bump_shrink_probability': 1, 'bump_shrink_scale_range': [.5, .5]})
    assert 0 < (b == 4).sum() < (labels == 4).sum()
    assert (b[:60] == 255).all()
    assert (b[:, :80] == 255).all()
    assert set(np.unique(b)) == {0, 1, 2, 4, 255}
    assert a.shape == image.shape and (a == 1).all()


def test_shrink_default_and_oversize_preserve_rng():
    from pinky_lane.augmentations import augment_bump_shrink
    labels = np.full((240, 320), 4, dtype=np.uint8)
    image = np.ones((240, 320, 3), dtype=np.float32)
    random.seed(17)
    expected = random.random()
    random.seed(17)
    a, b = augment_bump_shrink(image, labels, {'bump_shrink_probability': 1})
    assert a is image and b is labels
    assert random.random() == expected


def test_shrink_rejects_invalid_range_and_noisy_padding_labels():
    with pytest.raises(ValueError):
        validate({'bump_shrink_scale_range': [.01, .8]})


def test_shrink_falls_back_when_nearest_would_erase_tiny_positive():
    from pinky_lane.augmentations import augment_bump_shrink
    labels = np.zeros((240, 320), dtype=np.uint8)
    labels[131, 141] = 4  # Half-scale nearest samples even rows/columns.
    image = np.ones((240, 320, 3), dtype=np.float32)
    a, b = augment_bump_shrink(image, labels, {
        'bump_shrink_probability': 1, 'bump_shrink_scale_range': [.5, .5]})
    assert a is image and b is labels
    assert (b == 4).sum() == 1
