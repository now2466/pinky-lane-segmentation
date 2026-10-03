"""Legacy weighted CE and partial-label five-class CE/Dice."""

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import FULL_POLICY, LEGACY_POLICY


def class_weights(pixel_counts):
    counts = np.asarray(pixel_counts)
    if counts.shape != (4,) or np.any(counts <= 0):
        raise ValueError("Four-class training requires pixels from every class")
    return np.sqrt(counts.max() / counts).clip(1, 6).astype(np.float32)


def validate_batch(logits, labels, policies):
    if logits.ndim != 4 or logits.shape[1] not in (4, 5):
        raise ValueError("Expected [N,4|5,H,W] logits")
    if labels.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError("Labels must have shape [N,H,W]")
    if len(policies) != len(labels):
        raise ValueError("One policy is required per sample")
    observed = set(torch.unique(labels).tolist())
    if not observed <= set(range(logits.shape[1])) | {255}:
        raise ValueError("Invalid class IDs in labels")
    if not torch.any(labels != 255):
        raise ValueError("Batch contains no supervised pixels")


def partial_label_cross_entropy(logits, labels, policies):
    validate_batch(logits, labels, policies)
    if logits.shape[1] != 5:
        raise ValueError("Partial supervision requires five-class logits")
    log_probs = F.log_softmax(logits, dim=1)
    losses = []
    valid_count = 0
    for i, policy in enumerate(policies):
        target = labels[i]
        if policy not in {LEGACY_POLICY, FULL_POLICY}:
            raise ValueError("Invalid five-class supervision policy")
        supervised = target.clone()
        if policy == LEGACY_POLICY:
            if torch.any(target == 4):
                raise ValueError("Legacy partial-label masks cannot contain class 4")
            supervised[target == 0] = 255
        loss_map = F.nll_loss(log_probs[i:i + 1], supervised.unsqueeze(0),
                              ignore_index=255, reduction="none").squeeze(0)
        if policy == LEGACY_POLICY:
            marginal = -torch.logaddexp(log_probs[i, 0], log_probs[i, 4])
            loss_map = torch.where(target == 0, marginal, loss_map)
        valid = target != 255
        losses.append(loss_map[valid].sum())
        valid_count += int(valid.sum().item())
    return torch.stack(losses).sum() / valid_count


def masked_dice_loss(logits, labels, policies, smooth=1.0):
    validate_batch(logits, labels, policies)
    probs = torch.softmax(logits, dim=1)
    scores = []
    for class_id in range(1, logits.shape[1]):
        eligible = torch.tensor([class_id != 4 or policy == FULL_POLICY for policy in policies],
                                device=labels.device, dtype=torch.bool)
        valid = (labels != 255) & eligible[:, None, None]
        if not torch.any(valid):
            continue
        target = (labels == class_id) & valid
        probability = probs[:, class_id] * valid
        intersection = (probability * target).sum()
        scores.append((2 * intersection + smooth) / (probability.sum() + target.sum() + smooth))
    if not scores:
        raise ValueError("No Dice-supervised pixels")
    return 1 - torch.stack(scores).mean()


def segmentation_loss(logits, labels, policies, weights=None, dice_weight=0.3):
    if logits.shape[1] == 4:
        validate_batch(logits, labels, policies)
        ce = F.cross_entropy(logits, labels, weight=weights, ignore_index=255)
    else:
        ce = partial_label_cross_entropy(logits, labels, policies)
    return ce + dice_weight * masked_dice_loss(logits, labels, policies)
