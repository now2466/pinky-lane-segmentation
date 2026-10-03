import json

import pytest

from pinky_lane.checkpoints import load_checkpoint, load_model
from pinky_lane.models import ARCHITECTURES, CLASSES
from pinky_lane.train import run
from .helpers import synthetic_dataset


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
