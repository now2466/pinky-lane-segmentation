"""Rank supervised segmentation errors for review; never change labels or splits."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .checkpoints import load_model, sha256
from .dataset import CONFIRMED_BACKGROUND, FULL_POLICY, LaneDataset, sample_id
from .inference import overlay, preprocess


def error_counts(prediction, labels, policy):
    valid = labels != 255
    target = labels.copy()
    target[target == CONFIRMED_BACKGROUND] = 0
    # Legacy class-0 bump status is unknown. Do not call its bump predictions FP.
    known_bump = valid & ((labels != 0) | (policy == FULL_POLICY))
    mismatch = valid & (prediction != target)
    mismatch &= ~((prediction == 4) & ~known_bump)
    foreground = (target != 0) | (prediction != 0)
    lane_swap = valid & (((target == 1) & (prediction == 2)) |
                        ((target == 2) & (prediction == 1)))
    bump_fp = known_bump & (prediction == 4) & (target != 4)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bump_fp.astype(np.uint8), 8)
    return {'foreground_error_pixels': int((mismatch & foreground).sum()),
            'lane_role_confusion_pixels': int(lane_swap.sum()),
            'speed_bump_false_positive_pixels': int(bump_fp.sum()),
            'speed_bump_false_negative_pixels': int((valid & (target == 4) & (prediction != 4)).sum()),
            'speed_bump_fp_components': int(n-1),
            'small_bump_fp_components': int(sum(stats[i, cv2.CC_STAT_AREA] <= 48 for i in range(1, n)))}


@torch.inference_mode()
def run(model_path, dataset_root, split, output, device_name='auto', limit=48):
    output = Path(output)
    if output.exists() or limit < 1:
        raise ValueError('New output directory and positive limit required')
    model, device, count, _ = load_model(model_path, device_name, ignore_top=0)
    torch.set_num_threads(4)
    dataset = LaneDataset(dataset_root, split, count)
    rows = []
    for row in dataset.rows:
        image, labels, policy = dataset.read(row)
        prediction = model(preprocess(image).unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy().astype(np.uint8)
        detail = error_counts(prediction, labels, policy)
        rows.append({'sample_id': sample_id(row), 'split': split, 'policy': policy,
                     'image_sha256': sha256(dataset.paths(row)[0]), **detail})
    # Different categories get their own list: pixel count must not bury tiny FP.
    categories = ('foreground_error_pixels', 'lane_role_confusion_pixels',
                  'speed_bump_false_positive_pixels', 'speed_bump_false_negative_pixels',
                  'small_bump_fp_components')
    top = {key: [r['sample_id'] for r in sorted(rows, key=lambda r: (-r[key], r['sample_id']))[:limit]
                 if r[key] > 0] for key in categories}
    selected = set(name for group in top.values() for name in group)
    output.mkdir(parents=True)
    (output / 'cases').mkdir()
    for row in dataset.rows:
        name = sample_id(row)
        if name not in selected:
            continue
        image, labels, _ = dataset.read(row)
        predicted = model(preprocess(image).unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy().astype(np.uint8)
        predicted[labels == 255] = 255
        target = labels.copy()
        target[target == CONFIRMED_BACKGROUND] = 0
        panels = [image, overlay(image, target), overlay(image, predicted)]
        for panel, title in zip(panels, ('ORIGINAL', 'ANNOTATION (PARTIAL IF LEGACY)', 'PREDICTION')):
            cv2.putText(panel, title, (5, 15), 0, .35, (255, 255, 255), 1)
        if not cv2.imwrite(str(output / 'cases' / (name + '.jpg')), np.concatenate(panels, axis=1)):
            raise OSError('Cannot write case image')
    report = {'split': split, 'evaluated_frames': len(rows), 'model_sha256': sha256(model_path),
              'dataset_manifest_sha256': sha256(Path(dataset_root) / 'manifest.json'),
              'note': 'Diagnostic only. Legacy unknown background excluded from bump FP. No labels or splits modified; validation/test cases must not be added to train.',
              'top_by_category': top, 'frames': rows}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='val')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--limit', type=int, default=48)
    args = parser.parse_args()
    report = run(args.model, args.dataset, args.split, args.output, args.device, args.limit)
    print(json.dumps({'evaluated_frames': report['evaluated_frames'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
