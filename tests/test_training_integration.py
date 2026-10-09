import json

import pytest

from pinky_lane.checkpoints import load_checkpoint, load_model
from pinky_lane.models import ARCHITECTURES, CLASSES
from pinky_lane.train import run
from .helpers import make_checkpoint, synthetic_dataset


@pytest.mark.parametrize("num_classes", [4, 5])
def test_one_cpu_epoch_on_generated_data_writes_loadable_checkpoints(tmp_path, num_classes):
    dataset_root = synthetic_dataset(tmp_path / "synthetic-data", num_classes)
    output = tmp_path / f"run-{num_classes}"
    config = {
        "num_classes": num_classes,
        "training": {"batch_size": 1, "workers": 0, "epochs": 1,
                     "learning_rate": 0.0001, "cpu_threads": 1,
                     "amp": False, "seed": 7},
        "augmentation": {"lighting_probability": 0},
        "inference": {"ignore_top": 0},
    }

    result = run(config, dataset_root, output=output, device_name="cpu", epochs=1)

    assert result["training_started"] is True
    assert result["epochs_completed"] == 1
    assert (output / "provenance.json").is_file()
    assert (output / "metrics.jsonl").is_file()
    for filename in ("best.pt", "last.pt"):
        checkpoint, count = load_checkpoint(output / filename)
        assert count == num_classes
        assert checkpoint["architecture"] == ARCHITECTURES[num_classes]
        assert tuple(checkpoint["classes"]) == CLASSES[:num_classes]
        assert checkpoint["inference"]["ignore_top"] == 0
        model, device, loaded_count, ignore_top = load_model(output / filename, "cpu")
        assert model.training is False
        assert str(device) == "cpu"
        assert (loaded_count, ignore_top) == (num_classes, 0)

    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["items"]) == 4


def test_warm_start_guard_preserves_baseline_and_early_stops(tmp_path, monkeypatch):
    import torch
    from pinky_lane.metrics import fixed_foreground_metrics
    from pinky_lane.dataset import FULL_POLICY
    root = synthetic_dataset(tmp_path / 'data', 5)
    warm = tmp_path / 'warm.pt'
    original = make_checkpoint(warm, 5, {'ignore_top': 0})
    baseline = fixed_foreground_metrics({FULL_POLICY: torch.eye(5, dtype=torch.int64)*100}, 5)
    matrix = torch.eye(5, dtype=torch.int64)*100
    matrix[1, 1] = 50
    matrix[1, 2] = 50
    worse = fixed_foreground_metrics({FULL_POLICY: matrix}, 5)
    reports = iter([{'fixed_foreground': baseline}, {'fixed_foreground': worse}])
    monkeypatch.setattr('pinky_lane.train.evaluate', lambda *args: next(reports))
    config = {'num_classes': 5, 'inference': {'ignore_top': 0},
              'augmentation': {'lighting_probability': 0},
              'training': {'batch_size': 1, 'workers': 0, 'epochs': 3,
                           'learning_rate': .00003, 'cpu_threads': 1, 'amp': False,
                           'selection_metric': 'fixed_foreground',
                           'baseline_guard_max_iou_drop': .01,
                           'early_stopping_patience': 1, 'lr_scheduler': 'plateau',
                           'lr_scheduler_patience': 0, 'confirmed_background_dice': False,
                           'save_best_trained': True}}
    output = tmp_path / 'guarded'
    result = run(config, root, output, warm, 'cpu')
    assert result['epochs_completed'] == 1 and result['early_stopped']
    best, _ = load_checkpoint(output / 'best.pt')
    assert best['epoch'] == 0
    for key, value in original.state_dict().items():
        torch.testing.assert_close(best['model_state'][key], value, rtol=0, atol=0)
    record = json.loads((output / 'metrics.jsonl').read_text())
    assert not record['checkpoint_accepted']
    assert 'lane_left' in record['baseline_guard_regressions']
    assert record['next_learning_rate'] == .000015
    assert not (output / 'best-trained.pt').exists()
    experimental, _ = load_checkpoint(output / 'best-experimental.pt')
    assert experimental['epoch'] == 1
    assert experimental['baseline_guard_regressions'] == ['lane_left', 'lane_right']


def test_best_trained_is_separate_from_unchanged_baseline(tmp_path, monkeypatch):
    import torch
    from pinky_lane.metrics import fixed_foreground_metrics
    from pinky_lane.dataset import FULL_POLICY
    root = synthetic_dataset(tmp_path / 'data', 5)
    warm = tmp_path / 'warm.pt'
    make_checkpoint(warm, 5, {'ignore_top': 0})
    original = torch.eye(5, dtype=torch.int64) * 1000
    slight_loss = original.clone()
    slight_loss[4, 4] -= 1
    slight_loss[4, 0] += 1
    reports = iter([{'fixed_foreground': fixed_foreground_metrics({FULL_POLICY: matrix}, 5)}
                    for matrix in (original, slight_loss)])
    monkeypatch.setattr('pinky_lane.train.evaluate', lambda *args: next(reports))
    config = {'num_classes': 5, 'inference': {'ignore_top': 0},
              'augmentation': {'lighting_probability': 0},
              'training': {'batch_size': 1, 'workers': 0, 'epochs': 1,
                           'learning_rate': .00003, 'cpu_threads': 1, 'amp': False,
                           'selection_metric': 'fixed_foreground',
                           'baseline_guard_max_iou_drop': .01, 'save_best_trained': True,
                           'bump_positive_weight': .05}}
    output = tmp_path / 'guarded'
    result = run(config, root, output, warm, 'cpu')
    baseline, _ = load_checkpoint(output / 'best.pt')
    trained, _ = load_checkpoint(output / 'best-trained.pt')
    assert baseline['epoch'] == 0
    assert trained['epoch'] == 1
    assert result['best_trained_score'] < result['best_score']
