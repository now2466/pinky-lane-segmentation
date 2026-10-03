import json

import cv2
import numpy as np
import torch

from pinky_lane.dataset import FULL_POLICY, LEGACY_POLICY
from pinky_lane.models import ARCHITECTURES, CLASSES, LaneUNet


V11_SCHEMA = "fiveclass-user-reviewed-export-v11-proportional-speedbump-split-v1"


def synthetic_dataset(root, num_classes):
    """Create a four sample v11 dataset entirely under a test temporary directory."""
    counts = {"train": 2, "val": 1, "test": 1}
    items = []
    ordinal = 0
    for split in ("train", "val", "test"):
        for index in range(counts[split]):
            sample_id = f"{split}_{index}"
            if num_classes == 5:
                policy = (LEGACY_POLICY if (split == "test" or
                                             (split == "train" and index == 0))
                          else FULL_POLICY)
                label_count = 4 if policy == LEGACY_POLICY else 5
            else:
                policy = None
                label_count = 4

            x = (np.arange(320, dtype=np.uint16) * label_count // 320).astype(np.uint8)
            labels = np.broadcast_to(x, (240, 320)).copy()
            image = np.empty((240, 320, 3), dtype=np.uint8)
            image[:, :, 0] = (np.arange(320, dtype=np.uint16)[None, :] +
                              19 * ordinal) % 256
            image[:, :, 1] = (np.arange(240, dtype=np.uint16)[:, None] * 3 +
                              31 * ordinal) % 256
            image[:, :, 2] = (73 + 23 * ordinal) % 256

            image_dir = root / split / "images"
            mask_dir = root / split / "masks"
            image_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(image_dir / f"{sample_id}.jpg"), image)
            assert cv2.imwrite(str(mask_dir / f"{sample_id}.png"), labels)

            row = {"sample_id": sample_id, "split": split,
                   "source_group": f"synthetic-{split}-{index}"}
            if policy is not None:
                row["supervision_policy"] = policy
            items.append(row)
            ordinal += 1

    root.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": V11_SCHEMA, "counts": counts, "items": items}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def make_checkpoint(path, num_classes, inference=None):
    model = LaneUNet(num_classes)
    payload = {"architecture": ARCHITECTURES[num_classes],
               "classes": CLASSES[:num_classes], "model_state": model.state_dict()}
    if inference is not None:
        payload["inference"] = inference
    torch.save(payload, path)
    return model
