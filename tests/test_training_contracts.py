import pytest
import torch

from pinky_lane.checkpoints import load_checkpoint, load_model, sha256, warm_start
from pinky_lane.dataset import FULL_POLICY, LEGACY_POLICY
from pinky_lane.losses import masked_dice_loss, partial_label_cross_entropy
from pinky_lane.models import ARCHITECTURES, CLASSES, LaneUNet
from .helpers import make_checkpoint


def test_legacy_partial_ce_marginalizes_background_and_ignores_255():
    logits = torch.tensor([
        [[[2.0, 0.3, 500.0, -500.0]], [[0.1, 1.4, -500.0, 500.0]],
         [[-0.2, 0.2, 1.0, 0.0]], [[0.4, -0.4, 2.0, 0.0]],
         [[1.5, 0.0, 0.5, 1.0]]],
        [[[1.0, 0.0, -500.0, 500.0]], [[-0.3, 0.2, 500.0, -500.0]],
         [[0.1, -0.1, 2.0, 0.0]], [[0.0, 0.3, 1.0, 0.0]],
         [[0.5, 1.2, -2.0, 1.0]]],
    ])
    labels = torch.tensor([[[0, 1, 255, 255]], [[0, 4, 255, 255]]])

    actual = partial_label_cross_entropy(logits, labels, [LEGACY_POLICY, FULL_POLICY])
    log_probs = torch.log_softmax(logits, dim=1)
    expected = -torch.stack((
        torch.logaddexp(log_probs[0, 0, 0, 0], log_probs[0, 4, 0, 0]),
        log_probs[0, 1, 0, 1],
        log_probs[1, 0, 0, 0],
        log_probs[1, 4, 0, 1],
    )).sum() / 4
    torch.testing.assert_close(actual, expected)

    changed_ignored_logits = logits.clone()
    changed_ignored_logits[:, :, :, 2:] *= -7
    torch.testing.assert_close(
        partial_label_cross_entropy(changed_ignored_logits, labels,
                                    [LEGACY_POLICY, FULL_POLICY]), actual)


def test_legacy_dice_omits_class_four_supervision():
    logits = torch.tensor([[[[0.2, 1.1, 4.0]], [[1.0, 0.1, -2.0]],
                            [[-0.3, 0.6, 0.0]], [[0.1, -0.5, 1.0]],
                            [[2.5, 3.0, 8.0]]]])
    labels = torch.tensor([[[1, 2, 255]]])
    actual = masked_dice_loss(logits, labels, [LEGACY_POLICY])

    probabilities = torch.softmax(logits, dim=1)
    valid = labels != 255
    scores = []
    for class_id in (1, 2, 3):
        target = (labels == class_id) & valid
        probability = probabilities[:, class_id] * valid
        intersection = (probability * target).sum()
        scores.append((2 * intersection + 1) /
                      (probability.sum() + target.sum() + 1))
    expected = 1 - torch.stack(scores).mean()
    torch.testing.assert_close(actual, expected)


def test_four_to_five_warm_start_preserves_legacy_outputs_and_new_head(tmp_path):
    path = tmp_path / "generated-four-class.pt"
    source = make_checkpoint(path, 4).eval()
    target = LaneUNet(5).eval()
    new_weight = target.head.weight[4].detach().clone()
    new_bias = target.head.bias[4].detach().clone()

    assert warm_start(target, path) == sha256(path)
    for key, value in source.state_dict().items():
        if key not in {"head.weight", "head.bias"}:
            torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(target.head.weight[:4], source.head.weight, rtol=0, atol=0)
    torch.testing.assert_close(target.head.bias[:4], source.head.bias, rtol=0, atol=0)
    torch.testing.assert_close(target.head.weight[4], new_weight, rtol=0, atol=0)
    torch.testing.assert_close(target.head.bias[4], new_bias, rtol=0, atol=0)

    image = torch.rand(1, 3, 64, 80)
    with torch.inference_mode():
        old_logits = source(image)
        new_logits = target(image)
    torch.testing.assert_close(new_logits[:, :4], old_logits, rtol=0, atol=0)


def test_checkpoint_metadata_rejects_class_order_and_requires_five_class_roi(tmp_path):
    mismatched = tmp_path / "wrong-class-order.pt"
    model = LaneUNet(5)
    torch.save({"architecture": ARCHITECTURES[5],
                "classes": (CLASSES[0], CLASSES[2], CLASSES[1], *CLASSES[3:]),
                "model_state": model.state_dict()}, mismatched)
    with pytest.raises(ValueError, match="class order"):
        load_checkpoint(mismatched)

    five_class = tmp_path / "five-class.pt"
    make_checkpoint(five_class, 5)
    with pytest.raises(ValueError, match="requires --ignore-top"):
        load_model(five_class, "cpu")
    _, device, count, ignore_top = load_model(five_class, "cpu", ignore_top=42)
    assert str(device) == "cpu"
    assert (count, ignore_top) == (5, 42)

    four_class = tmp_path / "four-class.pt"
    make_checkpoint(four_class, 4)
    _, _, count, ignore_top = load_model(four_class, "cpu")
    assert (count, ignore_top) == (4, 110)
