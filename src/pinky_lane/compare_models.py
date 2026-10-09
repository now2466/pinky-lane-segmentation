"""Compare supervised speed-bump errors without changing labels or split membership."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .checkpoints import load_model, sha256
from .dataset import FULL_POLICY, LaneDataset, sample_id
from .inference import overlay, preprocess


def bump_counts(prediction, labels, policy):
    """Do not count bump predictions over unknown legacy background as false positives."""
    known = (labels != 255) & ((labels != 0) | (policy == FULL_POLICY))
    target = labels == 4
    tp = int(((prediction == 4) & target).sum())
    fp = int(((prediction == 4) & ~target & known).sum())
    fn = int(((prediction != 4) & target).sum())
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'iou': tp / (tp + fp + fn) if tp + fp + fn else None,
            'recall': tp / (tp + fn) if tp + fn else None}


@torch.inference_mode()
def run(reference, candidate, dataset_root, output, splits=('val',), device_name='auto'):
    output = Path(output)
    if output.exists():
        raise ValueError('A new output directory is required')
    torch.set_num_threads(4)
    models = {}
    for name, path in [('reference', reference), ('candidate', candidate)]:
        model, device, count, _ = load_model(path, device_name, ignore_top=0)
        if count != 5:
            raise ValueError('Speed-bump comparison requires five-class models')
        models[name] = model
    output.mkdir(parents=True)
    (output / 'cases').mkdir()
    rows = []
    for split in splits:
        dataset = LaneDataset(dataset_root, split, 5)
        for row in dataset.rows:
            image, labels, policy = dataset.read(row)
            target = labels == 4
            if not target.any():
                continue
            tensor = preprocess(image).unsqueeze(0).to(device)
            predictions = {name: model(tensor).argmax(1)[0].cpu().numpy().astype(np.uint8)
                           for name, model in models.items()}
            record = {'sample_id': sample_id(row), 'split': split,
                      'gt_pixels': int(target.sum()),
                      'gt_mean_bgr': float(image[target].mean()),
                      'gt_centroid_y': float(np.where(target)[0].mean()),
                      'source_group': row.get('source_group', row.get('source_video_sha256')),
                      **{name: bump_counts(prediction, labels, policy)
                         for name, prediction in predictions.items()}}
            record['iou_delta'] = record['candidate']['iou'] - record['reference']['iou']
            rows.append(record)
            save_case = (record['iou_delta'] < -.1 or record['gt_pixels'] <= 48
                         or (record['gt_pixels'] <= 512 and record['iou_delta'] > .1)
                         or record['candidate']['fp'] - record['reference']['fp'] > 100)
            if split != 'train' and save_case:
                display = {name: prediction.copy() for name, prediction in predictions.items()}
                for prediction in display.values():
                    prediction[labels == 255] = 255
                panels = [image, overlay(image, labels),
                          overlay(image, display['reference']),
                          overlay(image, display['candidate'])]
                path = output / 'cases' / f'{split}-{sample_id(row)}.jpg'
                if not cv2.imwrite(str(path), np.concatenate(panels, axis=1)):
                    raise OSError(f'Cannot write {path}')
    summary = {}
    for split in splits:
        selected = [r for r in rows if r['split'] == split]
        summary[split] = {'positive_frames': len(selected),
                          'regressed_frames_over_10pp': sum(r['iou_delta'] < -.1 for r in selected)}
        for lower, upper in [(1, 48), (49, 512), (513, 76800)]:
            group = [r for r in selected if lower <= r['gt_pixels'] <= upper]
            summary[split][f'area_{lower}_{upper}'] = {'frames': len(group)}
            for name in models:
                counts = {key: sum(r[name][key] for r in group) for key in ('tp', 'fp', 'fn')}
                counts['recall'] = counts['tp'] / (counts['tp'] + counts['fn']) if counts['tp'] + counts['fn'] else None
                summary[split][f'area_{lower}_{upper}'][name] = counts
    report = {'reference_sha256': sha256(reference), 'candidate_sha256': sha256(candidate),
              'manifest_sha256': sha256(Path(dataset_root) / 'manifest.json'),
              'ignore_top': 0, 'case_panel_order': ['image', 'annotation', 'reference', 'candidate'],
              'case_selection': 'All tiny positive frames, >10pp regression, >10pp small/medium improvement, or >100 extra FP pixels. Prediction displays suppress GT-ignore pixels; metrics unchanged.',
              'note': 'Area bins use total annotated bump pixels per positive frame. These are diagnostics, not full-class IoU; negative frames are excluded. Split membership unchanged.',
              'summary': summary, 'frames': rows}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', choices=('train', 'val', 'test'), default=['val'])
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    print(json.dumps(run(args.reference, args.candidate, args.dataset, args.output,
                         args.splits, args.device)['summary'], indent=2))


if __name__ == '__main__':
    main()
