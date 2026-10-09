"""Legacy weighted CE and partial-label five-class CE/Dice."""

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import CONFIRMED_BACKGROUND, FULL_POLICY, LEGACY_POLICY


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
    allowed = set(range(logits.shape[1])) | {255}
    if logits.shape[1] == 5:
        allowed.add(CONFIRMED_BACKGROUND)
    if not observed <= allowed:
        raise ValueError("Invalid class IDs in labels")
    for target, policy in zip(labels, policies):
        if torch.any(target == CONFIRMED_BACKGROUND) and policy != LEGACY_POLICY:
            raise ValueError("Confirmed background requires legacy partial supervision")
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
        supervised[target == CONFIRMED_BACKGROUND] = 0
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


def masked_dice_loss(logits, labels, policies, smooth=1.0, confirmed_background_dice=True):
    validate_batch(logits, labels, policies)
    probs = torch.softmax(logits, dim=1)
    scores = []
    for class_id in range(1, logits.shape[1]):
        eligible = torch.tensor([class_id != 4 or policy == FULL_POLICY for policy in policies],
                                device=labels.device, dtype=torch.bool)
        valid = (labels != 255) & eligible[:, None, None]
        if class_id == 4 and confirmed_background_dice:
            valid = valid | (labels == CONFIRMED_BACKGROUND)
        if not torch.any(valid):
            continue
        target = (labels == class_id) & valid
        probability = probs[:, class_id] * valid
        intersection = (probability * target).sum()
        scores.append((2 * intersection + smooth) / (probability.sum() + target.sum() + smooth))
    if not scores:
        raise ValueError("No Dice-supervised pixels")
    return 1 - torch.stack(scores).mean()


def segmentation_loss(logits, labels, policies, weights=None, dice_weight=0.3,
                      confirmed_background_dice=True):
    if logits.shape[1] == 4:
        validate_batch(logits, labels, policies)
        ce = F.cross_entropy(logits, labels, weight=weights, ignore_index=255)
    else:
        ce = partial_label_cross_entropy(logits, labels, policies)
    return ce + dice_weight * masked_dice_loss(
        logits, labels, policies, confirmed_background_dice=confirmed_background_dice)


def lane_distillation_loss(logits, teacher_logits, labels, confidence=0.75, temperature=2.0):
    """Anchor only confident teacher pixels agreeing with annotated lanes/crosswalk.

    Never imitate unknown background, bumps, ignored pixels, or teacher errors.
    Teacher is detached even if called incorrectly with gradient-enabled tensors.
    """
    if logits.shape != teacher_logits.shape or labels.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError("Distillation tensor shapes differ")
    if not 0 <= confidence <= 1 or temperature <= 0:
        raise ValueError("Invalid distillation settings")
    teacher_logits = teacher_logits.detach()
    certainty, predicted = teacher_logits.softmax(1).max(1)
    selected = (labels >= 1) & (labels <= 3) & (predicted == labels) & (certainty >= confidence)
    if not torch.any(selected):
        return logits.sum() * 0
    divergence = F.kl_div(F.log_softmax(logits / temperature, dim=1),
                          F.softmax(teacher_logits / temperature, dim=1), reduction="none").sum(1)
    return divergence[selected].mean() * temperature ** 2


def bump_positive_loss(logits, labels, policies, small_max_pixels=512, small_weight=2.0):
    """Equal-per-frame positive CE; never supervise unknown legacy bump status.

    Real tiny bumps otherwise contribute very few pixels to batch CE. This term
    gives each fully labeled bump-positive frame a vote, with optional rare-small
    emphasis. It does not label missing pixels or imitate a teacher's mistakes.
    """
    validate_batch(logits, labels, policies)
    if logits.shape[1] != 5 or small_max_pixels < 1 or small_weight < 1:
        raise ValueError("Invalid bump preservation settings")
    terms = []
    log_probability = F.log_softmax(logits, dim=1)[:, 4]
    for i, policy in enumerate(policies):
        positive = labels[i] == 4
        if policy != FULL_POLICY or not torch.any(positive):
            continue
        weight = small_weight if int(positive.sum()) <= small_max_pixels else 1.0
        terms.append(-log_probability[i][positive].mean() * weight)
    return torch.stack(terms).mean() if terms else logits.sum() * 0
