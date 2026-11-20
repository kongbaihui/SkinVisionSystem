import numpy as np
import pytest
import torch

from backend.skinvision.checkpoints import load_model_weights
from backend.skinvision.models.Encoder import SS2D
from backend.skinvision.training.loader import dataset_normalized


def test_strict_checkpoint_loading(tmp_path):
    model = torch.nn.Linear(2, 1)
    path = tmp_path / 'checkpoint.pth'
    torch.save({'model_state_dict': {'module.' + k: v for k, v in model.state_dict().items()}}, path)
    restored = torch.nn.Linear(2, 1)
    load_model_weights(restored, path)
    torch.testing.assert_close(restored.weight, model.weight)
    torch.save({'weight': model.weight}, path)
    with pytest.raises(RuntimeError, match='Missing key'):
        load_model_weights(restored, path)


def test_ss2d_cpu_forward_and_backward():
    torch.set_num_threads(2)
    layer = SS2D(d_model=4, d_state=2)
    x = torch.randn(2, 4, 4, 4, requires_grad=True)
    y = layer(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert layer.A_logs.grad is not None


def test_constant_dataset_normalization_is_finite():
    assert np.isfinite(dataset_normalized(np.ones((2, 4, 4, 3)))).all()


def test_legacy_training_checkpoint_with_numpy_loss(tmp_path):
    model = torch.nn.Linear(2, 1)
    path = tmp_path / 'latest.pth'
    torch.save({'model_state_dict': model.state_dict(), 'loss': np.float64(0.5)}, path)
    load_model_weights(model, path)


def test_only_known_profiling_counters_are_removed(tmp_path):
    model = torch.nn.Linear(2, 1)
    path = tmp_path / 'profiled.pth'
    state = {**model.state_dict(), 'total_ops': torch.tensor([10.], dtype=torch.double)}
    torch.save(state, path)
    with pytest.warns(UserWarning, match='profiling'):
        load_model_weights(model, path)
    assert model.checkpoint_info['removed_profiling_keys'] == ['total_ops']
    state['unknown.total_ops'] = torch.tensor([1.])
    torch.save(state, path)
    with pytest.raises(RuntimeError, match='Unexpected key'):
        load_model_weights(model, path)
