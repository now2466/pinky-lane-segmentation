import cv2
import numpy as np
import pytest

from pinky_lane.dataset import (FOUR_POLICY, FULL_POLICY, LEGACY_POLICY, LaneDataset,
                                check_dataset)
from .helpers import V11_SCHEMA, synthetic_dataset


def test_v11_manifest_reads_four_and_five_class_supervision_policies(tmp_path):
    root = synthetic_dataset(tmp_path / "data", 5)
    dataset = LaneDataset(root, "train", 5)
    _, legacy_labels, legacy_policy, _ = dataset[0]
    _, full_labels, full_policy, _ = dataset[1]
    report = check_dataset(root, 5)

    assert dataset.manifest["schema_version"] == V11_SCHEMA
    assert str(legacy_labels.dtype) == "torch.int64"
    assert legacy_policy == LEGACY_POLICY
    assert 4 not in legacy_labels.unique().tolist()
    assert full_policy == FULL_POLICY
    assert 4 in full_labels.unique().tolist()
    assert report["policy_counts"]["train"] == {LEGACY_POLICY: 1, FULL_POLICY: 1}

    four_class = LaneDataset(synthetic_dataset(tmp_path / "four", 4), "train", 4)
    assert four_class[0][2] == FOUR_POLICY


@pytest.mark.parametrize("invalid_mask, message", [
    (np.zeros((240, 320), dtype=np.uint16), "uint8 mask"),
    (np.zeros((239, 320), dtype=np.uint8), "uint8 mask"),
    (np.full((240, 320), 255, dtype=np.uint8), "no supervised pixels"),
    (np.pad(np.array([[4]], dtype=np.uint8), ((0, 239), (0, 319))), "Invalid labels"),
])
def test_dataset_rejects_invalid_masks(tmp_path, invalid_mask, message):
    root = synthetic_dataset(tmp_path / "data", 5)
    mask_path = root / "train" / "masks" / "train_0.png"
    assert cv2.imwrite(str(mask_path), invalid_mask)
    dataset = LaneDataset(root, "train", 5)

    with pytest.raises(ValueError, match=message):
        dataset.read(dataset.rows[0])


def test_preflight_blocks_exact_image_duplicates_across_splits(tmp_path):
    root = synthetic_dataset(tmp_path / "data", 5)
    train_image = (root / "train" / "images" / "train_0.jpg").read_bytes()
    (root / "val" / "images" / "val_0.jpg").write_bytes(train_image)

    with pytest.raises(ValueError, match="Exact image duplicates across splits"):
        check_dataset(root, 5)
