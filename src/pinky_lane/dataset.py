"""Manifest-based loaders and read-only preflight, including reviewed v11."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentations import augment_bump_crop, augment_bump_shrink, augment_image, validate
from .checkpoints import sha256

LEGACY_POLICY = "legacy_partial_class4"
FULL_POLICY = "fully_labeled_5class"
FOUR_POLICY = "fully_labeled_4class"
# Training-only sentinel: confirmed background, not a sixth semantic class.
CONFIRMED_BACKGROUND = 254
SCHEMAS = {
    "pinky-lane-dataset-v1", "lr-training-export-v1", "lr-training-export-v2",
    "lr-training-export-v3", "lr-training-export-v4", "lr-training-export-v5-partial-label",
    "fiveclass-user-reviewed-export-v11-proportional-speedbump-split-v1",
}


def sample_id(row):
    name = row.get("sample_id")
    if name is None and "frame" in row:
        name = f"{int(row['frame']):06d}"
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("sample_id must be a nonempty filename stem without directories")
    if "/" in name or "\\" in name:
        raise ValueError("Invalid sample_id")
    return name


def supervision_policy(row, num_classes):
    policy = row.get("supervision_policy")
    if num_classes == 4:
        if policy not in {None, FOUR_POLICY, LEGACY_POLICY}:
            raise ValueError("Fully five-class rows require a five-class model")
        return FOUR_POLICY
    if policy not in {LEGACY_POLICY, FULL_POLICY}:
        raise ValueError("Five-class rows require an explicit supervision_policy")
    return policy


class LaneDataset(Dataset):
    def __init__(self, root, split, num_classes=5, augmentation=None):
        self.root = Path(root)
        self.split = split
        self.num_classes = num_classes
        if split not in {"train", "val", "test"} or num_classes not in {4, 5}:
            raise ValueError("Invalid split or class count")
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") not in SCHEMAS:
            raise ValueError(f"Unsupported manifest schema: {self.manifest.get('schema_version')}")
        self.rows = [row for row in self.manifest.get("items", []) if row.get("split") == split]
        if not self.rows:
            raise ValueError(f"Empty split: {split}")
        self.augmentation = augmentation
        if augmentation is not None:
            if split != "train":
                raise ValueError("Augmentation is allowed only on the training split")
            validate(augmentation)

    def __len__(self):
        return len(self.rows)

    def paths(self, row):
        name = sample_id(row)
        return (self.root / self.split / "images" / f"{name}.jpg",
                self.root / self.split / "masks" / f"{name}.png")

    def read(self, row):
        image_path, mask_path = self.paths(row)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        labels = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if bgr is None or bgr.shape != (240, 320, 3):
            raise ValueError(f"Expected a 320x240 RGB image: {image_path.name}")
        if labels is None or labels.shape != (240, 320) or labels.dtype != np.uint8:
            raise ValueError(f"Expected a 320x240 uint8 mask: {mask_path.name}")
        policy = supervision_policy(row, self.num_classes)
        count = 4 if policy == LEGACY_POLICY else self.num_classes
        if not set(np.unique(labels)) <= set(range(count)) | {255}:
            raise ValueError(f"Invalid labels for {policy}: {mask_path.name}")
        if not np.any(labels != 255):
            raise ValueError(f"Mask has no supervised pixels: {mask_path.name}")
        polygons = row.get("confirmed_background_polygons")
        if polygons is not None:
            if self.split != "train" or self.num_classes != 5 or policy != LEGACY_POLICY:
                raise ValueError("Confirmed background is only allowed in five-class legacy training")
            if row.get("image_sha256") != sha256(image_path) or row.get("mask_sha256") != sha256(mask_path):
                raise ValueError("Confirmed background requires matching image/mask hashes")
            if not isinstance(polygons, list) or not polygons:
                raise ValueError("Expected nonempty confirmed-background polygon list")
            region = np.zeros(labels.shape, np.uint8)
            for polygon in polygons:
                points = np.asarray(polygon)
                if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                        or not np.issubdtype(points.dtype, np.integer)
                        or np.any(points < 0) or np.any(points[:, 0] >= 320)
                        or np.any(points[:, 1] >= 240)):
                    raise ValueError("Polygon must contain >=3 integer (x,y) points inside 320x240")
                cv2.fillPoly(region, [points.astype(np.int32)], 1)
            # Never overwrite a lane, crosswalk, speed bump or ignored pixel.
            selected = (region != 0) & (labels == 0)
            if not np.any(selected):
                raise ValueError("Confirmed-background polygons contain no background pixels")
            labels[selected] = CONFIRMED_BACKGROUND
        return bgr, labels, policy

    def __getitem__(self, index):
        row = self.rows[index]
        bgr, labels, policy = self.read(row)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        if self.augmentation is not None:
            rgb, labels = augment_bump_shrink(rgb, labels, self.augmentation)
            rgb, labels = augment_bump_crop(rgb, labels, self.augmentation)
            rgb = augment_image(rgb, self.augmentation)
        group = row.get("source_video_sha256") or row.get("source_group") or "unspecified"
        return (torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))),
                torch.from_numpy(labels.astype(np.int64)), policy, str(group))


def check_dataset(root, num_classes=5):
    """Verify all files, optional supplied hashes, and exact cross-split image leakage."""
    counts = {}
    policy_counts = {}
    class_positive_counts = {}
    image_splits = defaultdict(set)
    source_splits = defaultdict(set)
    identities = set()
    manifest = None
    for split in ("train", "val", "test"):
        dataset = LaneDataset(root, split, num_classes)
        manifest = dataset.manifest
        counts[split] = len(dataset)
        policies = Counter()
        positives = Counter()
        for row in dataset.rows:
            identity = (split, sample_id(row))
            if identity in identities:
                raise ValueError(f"Duplicate manifest row: {identity}")
            identities.add(identity)
            _, labels, policy = dataset.read(row)
            labels = np.where(labels == CONFIRMED_BACKGROUND, 0, labels)
            policies[policy] += 1
            for class_id in np.unique(labels):
                if class_id != 255:
                    positives[str(int(class_id))] += 1
            image_path, mask_path = dataset.paths(row)
            image_hash = sha256(image_path)
            for key, actual in (("image_sha256", image_hash), ("mask_sha256", sha256(mask_path))):
                if row.get(key) is not None and row[key] != actual:
                    raise ValueError(f"Hash mismatch: {identity} {key}")
            image_splits[image_hash].add(split)
            group = row.get("source_video_sha256") or row.get("source_group")
            if group:
                source_splits[str(group)].add(split)
        policy_counts[split] = dict(policies)
        class_positive_counts[split] = dict(positives)
        image_files = {p.stem for p in (Path(root) / split / "images").glob("*.jpg")}
        mask_files = {p.stem for p in (Path(root) / split / "masks").glob("*.png")}
        names = {sample_id(row) for row in dataset.rows}
        if image_files != names or mask_files != names:
            raise ValueError(f"Manifest and file inventory mismatch: {split}")
    if manifest.get("counts") is not None and manifest["counts"] != counts:
        raise ValueError("Manifest split counts differ from actual active items")
    leaks = sum(len(splits) > 1 for splits in image_splits.values())
    if leaks:
        raise ValueError(f"Exact image duplicates across splits: {leaks}")
    return {"data_ready": True, "counts": counts, "policy_counts": policy_counts,
            "class_positive_frames": class_positive_counts,
            "exact_image_hash_cross_split_leaks": leaks,
            "source_groups_spanning_splits": sum(len(v) > 1 for v in source_splits.values()),
            "manifest_sha256": sha256(Path(root) / "manifest.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, choices=(4, 5), default=5)
    args = parser.parse_args()
    print(json.dumps(check_dataset(args.dataset, args.num_classes), indent=2))


if __name__ == "__main__":
    main()
