import json

import numpy as np
import pytest
import torch

from pinky_lane.checkpoints import sha256
from pinky_lane.dataset import CONFIRMED_BACKGROUND, FULL_POLICY, LEGACY_POLICY, LaneDataset, check_dataset
from pinky_lane.losses import masked_dice_loss, partial_label_cross_entropy, validate_batch
from tests.helpers import synthetic_dataset


def test_confirmed_background_penalizes_bump_but_legacy_stays_marginal():
    logits = torch.tensor([[[[0.]], [[-8.]], [[-8.]], [[-8.]], [[8.]]]], requires_grad=True)
    unknown = torch.zeros((1, 1, 1), dtype=torch.long)
    confirmed = torch.full_like(unknown, CONFIRMED_BACKGROUND)
    assert partial_label_cross_entropy(logits, unknown, [LEGACY_POLICY]) < 0.001
    loss = partial_label_cross_entropy(logits, confirmed, [LEGACY_POLICY])
    assert loss > 7.9
    loss.backward()
    assert logits.grad[0, 4, 0, 0] > 0.99
    assert logits.grad[0, 0, 0, 0] < -0.99
    assert masked_dice_loss(logits, confirmed, [LEGACY_POLICY]) > masked_dice_loss(logits, unknown, [LEGACY_POLICY])


def test_training_only_hash_bound_overlay_preserves_masks(tmp_path):
    root = synthetic_dataset(tmp_path, 5)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    row = manifest["items"][0]
    dataset = LaneDataset(root, "train")
    image, mask = dataset.paths(row)
    row.update(image_sha256=sha256(image), mask_sha256=sha256(mask),
               confirmed_background_polygons=[[[0, 100], [319, 100], [319, 239], [0, 239]]])
    path.write_text(json.dumps(manifest))
    _, before, _ = dataset.read(dataset.rows[0])
    _, after, _ = LaneDataset(root, "train").read(row)
    assert np.any(after == CONFIRMED_BACKGROUND)
    assert np.array_equal(after[before != 0], before[before != 0])
    assert sha256(mask) == row["mask_sha256"]
    audit = check_dataset(root)
    assert "254" not in audit["class_positive_frames"]["train"]
    row["image_sha256"] = "wrong"
    with pytest.raises(ValueError, match="hashes"):
        LaneDataset(root, "train").read(row)
    with pytest.raises(ValueError, match="only allowed"):
        LaneDataset(root, "train", 4).read(row)


def test_sentinel_rejected_for_four_classes_or_full_policy():
    labels = torch.full((1, 1, 1), CONFIRMED_BACKGROUND, dtype=torch.long)
    with pytest.raises(ValueError):
        validate_batch(torch.zeros(1, 4, 1, 1), labels, [LEGACY_POLICY])
    with pytest.raises(ValueError):
        validate_batch(torch.zeros(1, 5, 1, 1), labels, [FULL_POLICY])


def test_ce_only_ablation_preserves_ce_but_excludes_confirmed_bump_dice():
    from pinky_lane.losses import segmentation_loss
    logits = torch.zeros(2, 5, 1, 2)
    labels = torch.tensor([[[254, 1]], [[0, 4]]])
    policies = [LEGACY_POLICY, FULL_POLICY]
    ce = partial_label_cross_entropy(logits, labels, policies)
    expected = ce + .3 * masked_dice_loss(logits, labels, policies, confirmed_background_dice=False)
    actual = segmentation_loss(logits, labels, policies, confirmed_background_dice=False)
    torch.testing.assert_close(actual, expected)
    assert actual != segmentation_loss(logits, labels, policies, confirmed_background_dice=True)
