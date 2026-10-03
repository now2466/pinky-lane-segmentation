"""Keep fully labeled and legacy partial-label evaluation separate."""

from collections import defaultdict

import torch

from .dataset import LEGACY_POLICY
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


@torch.inference_mode()
def evaluate(model, loader, device, num_classes):
    model.eval()
    matrices = {}
    by_source = {}
    samples = defaultdict(int)
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
    return {"by_policy": {policy: {**summarize(matrix), "samples": samples[policy]}
                           for policy, matrix in matrices.items()},
            "by_source": [{"policy": policy, "source_group": group, **summarize(matrix)}
                          for (policy, group), matrix in sorted(by_source.items())]}


def selection_score(report, num_classes):
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
