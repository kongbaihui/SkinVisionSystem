"""One strict checkpoint-loading policy for serving, training and tools."""
import threading
import warnings
from codecs import encode
import numpy as np
import torch

_load_lock = threading.Lock()


def load_checkpoint(path, device='cpu'):
    # Old training logs stored np.mean loss scalars. Permit only their concrete
    # constructors, never unrestricted pickle loading or arbitrary model classes.
    allowed = [encode, np.core.multiarray.scalar, np.dtype, type(np.dtype(np.float64))]
    with _load_lock:
        previous = torch.serialization.get_safe_globals().copy()
        try:
            torch.serialization.add_safe_globals(allowed)
            return torch.load(path, map_location=device, weights_only=True)
        finally:
            torch.serialization.clear_safe_globals()
            torch.serialization.add_safe_globals(previous)


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError('Checkpoint must contain a state dictionary')
    for key in ('state_dict', 'model_state_dict', 'model'):
        if isinstance(checkpoint.get(key), dict):
            checkpoint = checkpoint[key]
            break
    if checkpoint and all(key.startswith('module.') for key in checkpoint):
        checkpoint = {key[7:]: value for key, value in checkpoint.items()}
    return checkpoint


def load_model_state(model, checkpoint):
    state = extract_state_dict(checkpoint).copy()
    expected = model.state_dict()
    modules = dict(model.named_modules())
    removed = []
    for key in list(state):
        module_name, _, counter = key.rpartition('.')
        if not module_name:
            counter = key
        value = state[key]
        if (key not in expected and module_name in modules and
                counter in {'total_ops', 'total_params'} and
                isinstance(value, torch.Tensor) and value.numel() == 1):
            removed.append(key)
            del state[key]
    if removed:
        warnings.warn(f'Removed {len(removed)} legacy THOP profiling counters; all model weights still load strictly.', stacklevel=2)
    model.load_state_dict(state, strict=True)
    model.checkpoint_info = {'strict': True, 'removed_profiling_keys': removed}
    return model


def load_model_weights(model, path, device='cpu'):
    return load_model_state(model, load_checkpoint(path, device))
