import json

import torch

from pinky_lane.checkpoints import load_checkpoint
from pinky_lane.losses import lane_distillation_loss
from pinky_lane.train import run
from .helpers import make_checkpoint, synthetic_dataset


def test_distillation_excludes_wrong_teacher_and_unknown_pixels():
    teacher = torch.full((1, 5, 1, 4), -5.0, requires_grad=True)
    with torch.no_grad():
        teacher[0, 1, 0] = 5
    student = torch.zeros_like(teacher, requires_grad=True)
    labels = torch.tensor([[[1, 2, 0, 255]]])
    loss = lane_distillation_loss(student, teacher, labels)
    loss.backward()
    assert loss > 0
    assert teacher.grad is None
    assert torch.any(student.grad[:, :, :, 0] != 0)
    assert torch.all(student.grad[:, :, :, 1:] == 0)
    empty = lane_distillation_loss(student, teacher, torch.zeros_like(labels))
    assert empty == 0 and empty.requires_grad


def test_head_only_freezes_weights_and_batchnorm_statistics(tmp_path):
    root = synthetic_dataset(tmp_path / 'data', 5)
    warm = tmp_path / 'warm.pt'
    source = make_checkpoint(warm, 5, {'ignore_top': 0})
    config = {'num_classes': 5, 'inference': {'ignore_top': 0},
              'augmentation': {'lighting_probability': 0},
              'training': {'batch_size': 1, 'workers': 0, 'epochs': 1,
                           'learning_rate': .0003, 'cpu_threads': 1, 'amp': False,
                           'selection_metric': 'fixed_foreground',
                           'trainable_modules': ['head'], 'lane_distillation_weight': .2}}
    output = tmp_path / 'output'
    result = run(config, root, output, warm, 'cpu')
    assert result['epochs_completed'] == 1
    last, _ = load_checkpoint(output / 'last.pt')
    for name, value in source.state_dict().items():
        if not name.startswith('head.'):
            torch.testing.assert_close(last['model_state'][name], value, rtol=0, atol=0)
    assert not torch.equal(last['model_state']['head.weight'], source.head.weight)
    assert json.loads((output / 'provenance.json').read_text())['trainable_modules'] == ['head']
