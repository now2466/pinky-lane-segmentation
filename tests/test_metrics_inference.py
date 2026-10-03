import numpy as np
import torch
from torch.utils.data import DataLoader

from pinky_lane.checkpoints import load_model
from pinky_lane.dataset import FULL_POLICY, LEGACY_POLICY, LaneDataset
from pinky_lane.export import export_model
from pinky_lane.inference import overlay, predict
from pinky_lane.metrics import confusion, evaluate, selection_score, summarize
from pinky_lane.train import run

from .helpers import make_checkpoint, synthetic_dataset


def test_legacy_speed_bump_predictions_are_unknown_background_not_evaluable_class_four():
    prediction = torch.tensor([[4, 1, 2, 3, 4]])
    target = torch.tensor([[0, 1, 2, 3, 255]])
    legacy = confusion(prediction, target, LEGACY_POLICY, 5)
    assert legacy.shape == (4, 4)
    torch.testing.assert_close(legacy, torch.eye(4, dtype=torch.int64))
    full = confusion(prediction, target, FULL_POLICY, 5)
    assert full.shape == (5, 5)
    assert full[0, 4] == 1
    assert full.sum() == 4


def test_absent_union_classes_are_null_and_policies_stay_separate(tmp_path):
    metrics = summarize(torch.diag(torch.tensor([3, 2, 0, 0, 0])))
    assert metrics["iou"]["speed_bump"] is None
    assert metrics["foreground_mean_iou"] == 1
    assert metrics["absent_union_classes"] == ["lane_right", "crosswalk", "speed_bump"]
    root = synthetic_dataset(tmp_path / "data", 5)
    model = make_checkpoint(tmp_path / "model.pt", 5).eval()
    report = evaluate(model, DataLoader(LaneDataset(root, "train", 5), batch_size=2),
                      torch.device("cpu"), 5)
    assert set(report["by_policy"]) == {LEGACY_POLICY, FULL_POLICY}
    assert len(report["by_policy"][LEGACY_POLICY]["confusion"]) == 4
    assert len(report["by_policy"][FULL_POLICY]["confusion"]) == 5
    expected = next(row["foreground_mean_iou"] for row in report["by_source"]
                    if row["policy"] == FULL_POLICY)
    assert selection_score(report, 5) == expected


def test_overlay_preserves_original_background_and_ignored_pixels():
    original = np.full((240, 320, 3), (12, 34, 56), dtype=np.uint8)
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[:50] = 255
    mask[100:120] = 4
    result = overlay(original, mask)
    np.testing.assert_array_equal(result[mask == 0], original[mask == 0])
    np.testing.assert_array_equal(result[mask == 255], original[mask == 255])
    assert np.any(result[mask == 4] != original[mask == 4])


def test_five_class_torchscript_export_retains_logits_and_roi(tmp_path):
    checkpoint = tmp_path / "five.pt"
    make_checkpoint(checkpoint, 5, {"ignore_top": 88})
    script_path = tmp_path / "model.torchscript.pt"
    metadata = export_model(checkpoint, script_path)
    eager, device, _, roi = load_model(checkpoint, "cpu")
    scripted, _, count, scripted_roi = load_model(script_path, "cpu", "torchscript")
    assert (count, roi, scripted_roi) == (5, 88, 88)
    assert metadata["reload_max_abs_error"] <= 1e-4
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    eager_mask = predict(eager, [frame], device, roi)[0]
    script_mask = predict(scripted, [frame], device, scripted_roi)[0]
    np.testing.assert_array_equal(eager_mask, script_mask)
    assert np.all(script_mask[:88] == 255)


def test_dry_run_does_not_create_training_outputs(tmp_path):
    root = synthetic_dataset(tmp_path / "data", 5)
    output = tmp_path / "must-not-be-created"
    config = {"num_classes": 5, "inference": {"ignore_top": 110},
              "augmentation": {"lighting_probability": 0},
              "training": {"epochs": 1, "batch_size": 1, "learning_rate": 0.0003,
                           "workers": 0, "cpu_threads": 1}}
    result = run(config, root, output, device_name="cpu", dry_run=True)
    assert result["training_started"] is False
    assert result["sample_loss_finite"] is True
    assert not output.exists()
