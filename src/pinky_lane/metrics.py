"""Keep fully labeled and legacy partial-label evaluation separate."""

from collections import defaultdict

import torch

from .dataset import FULL_POLICY, LEGACY_POLICY
from .models import CLASSES


def confusion(prediction, labels, policy, num_classes):
    valid = labels != 255
    if num_classes == 5 and policy == LEGACY_POLICY:
        num_classes = 4
        prediction = prediction.clone()
        prediction[prediction == 4] = 0
    indices = labels[valid] * num_classes + prediction[valid]
    return torch.bincount(indices, minlength=num_classes ** 2).view(num_classes, num_classes)


def summarize(matrix):
    matrix = matrix.to(torch.float64)
    diagonal = matrix.diag()
    union = matrix.sum(0) + matrix.sum(1) - diagonal
    ious = [float(diagonal[i] / union[i]) if union[i] else None for i in range(len(union))]
    present = [value for value in ious if value is not None]
    foreground = [value for value in ious[1:] if value is not None]
    return {
        "iou": dict(zip(CLASSES[:len(ious)], ious)),
        "mean_iou": sum(present) / len(present) if present else None,
        "foreground_mean_iou": sum(foreground) / len(foreground) if foreground else None,
        "pixel_accuracy": float(diagonal.sum() / matrix.sum()) if matrix.sum() else None,
        "absent_union_classes": [CLASSES[i] for i, value in enumerate(ious) if value is None],
        "confusion": matrix.to(torch.int64).tolist(),
    }


def fixed_foreground_metrics(matrices, num_classes):
    """Fixed GT-support classes; never evaluate unknown legacy bump background.

    Classes 1..3 use every annotated policy, class 4 uses FULL_POLICY only.
    Absent-GT class false positives remain explicit, not hidden in changing mIoU.
    This is class-macro pixel IoU, NOT the historical source-macro score.
    """
    result = {}
    included = []
    for class_id in range(1, num_classes):
        tp = fp = fn = valid_pixels = 0
        for policy, matrix in matrices.items():
            if class_id == 4 and policy != FULL_POLICY:
                continue
            matrix = torch.as_tensor(matrix, dtype=torch.int64)
            tp += int(matrix[class_id, class_id])
            fp += int(matrix[:, class_id].sum() - matrix[class_id, class_id])
            fn += int(matrix[class_id].sum() - matrix[class_id, class_id])
            valid_pixels += int(matrix.sum())
        support = tp + fn
        predicted = tp + fp
        union = tp + fp + fn
        iou = tp / union if union else None
        name = CLASSES[class_id]
        if support:
            included.append(name)
        negative_pixels = valid_pixels - support
        result[name] = {
            "iou": iou, "precision": tp / predicted if predicted else None,
            "recall": tp / support if support else None,
            "gt_pixels": support, "predicted_pixels": predicted,
            "true_positive_pixels": tp, "false_positive_pixels": fp,
            "false_negative_pixels": fn, "valid_pixels": valid_pixels,
            "false_positive_rate": fp / negative_pixels if negative_pixels else None,
        }
    lane_confusion = sum(int(torch.as_tensor(m)[1, 2] + torch.as_tensor(m)[2, 1])
                         for m in matrices.values())
    lane_support = sum(result[CLASSES[c]]["gt_pixels"] for c in (1, 2))
    return {"classes": result, "included_classes": included,
            "foreground_mean_iou": sum(result[c]["iou"] for c in included) / len(included)
            if included else None,
            "lane_role_confusion_pixels": lane_confusion,
            "lane_role_confusion_rate": lane_confusion / lane_support if lane_support else None,
            "metric_version": "fixed_gt_support_class_macro_v1"}


@torch.inference_mode()
def evaluate(model, loader, device, num_classes):
    model.eval()
    matrices = {}
    by_source = {}
    samples = defaultdict(int)
    size_matrices = {}
    size_samples = defaultdict(int)
    for images, labels, policies, groups in loader:
        predictions = model(images.to(device)).argmax(1).cpu()
        for i, policy in enumerate(policies):
            matrix = confusion(predictions[i], labels[i], policy, num_classes)
            if policy not in matrices:
                matrices[policy] = torch.zeros_like(matrix)
            matrices[policy] += matrix
            key = (policy, groups[i])
            if key not in by_source:
                by_source[key] = torch.zeros_like(matrix)
            by_source[key] += matrix
            samples[policy] += 1
            if num_classes == 5 and policy == FULL_POLICY:
                area = int((labels[i] == 4).sum())
                bucket = 'absent' if area == 0 else 'small_le_512' if area <= 512 else 'medium_le_2048' if area <= 2048 else 'large_gt_2048'
                if bucket not in size_matrices:
                    size_matrices[bucket] = torch.zeros_like(matrix)
                size_matrices[bucket] += matrix
                size_samples[bucket] += 1
    return {"fixed_foreground": fixed_foreground_metrics(matrices, num_classes),
            "by_bump_gt_size": {key: {**fixed_foreground_metrics({FULL_POLICY: matrix}, num_classes),
                                      "samples": size_samples[key]}
                                for key, matrix in size_matrices.items()},
            "by_policy": {policy: {**summarize(matrix), "samples": samples[policy]}
                           for policy, matrix in matrices.items()},
            "by_source": [{"policy": policy, "source_group": group, **summarize(matrix)}
                          for (policy, group), matrix in sorted(by_source.items())]}


def selection_score(report, num_classes, mode="legacy_source_macro"):
    if mode == "fixed_foreground":
        score = report["fixed_foreground"]["foreground_mean_iou"]
        if score is None:
            raise ValueError("No ground-truth foreground for checkpoint selection")
        return score
    if mode != "legacy_source_macro":
        raise ValueError("Unknown checkpoint selection mode")
    # Match five-class source-macro checkpoint selection; never mix class-4-unlabeled
    # old masks into the speed-bump validation score.
    if num_classes == 5:
        scores = [row["foreground_mean_iou"] for row in report["by_source"]
                  if row["policy"] == "fully_labeled_5class"
                  and row["foreground_mean_iou"] is not None]
        if not scores:
            raise ValueError("Five-class validation requires fully labeled five-class samples")
        return sum(scores) / len(scores)
    return report["by_policy"]["fully_labeled_4class"]["foreground_mean_iou"]
