"""Portable real-valued selective scan used by LMDAUNet.

Implements h[t] = exp(delta[t] A) h[t-1] + delta[t] B[t] u[t],
y[t] = C[t] h[t] + D u[t]. See Mamba v2.2.2 selective_scan_ref:
https://github.com/state-spaces/mamba/blob/v2.2.2/mamba_ssm/ops/selective_scan_interface.py
This is a native differentiable recurrence, not an image-processing fallback.
Complex states are not used by this model and are explicitly unsupported.
"""
import torch
from torch.nn import functional as F


def selective_scan_torch(u, delta, A, B, C, D=None, z=None,
                         delta_bias=None, delta_softplus=False,
                         return_last_state=False):
    if any(t.is_complex() for t in (u, delta, A, B, C)):
        raise ValueError("Only real selective-scan states are supported")
    if u.ndim != 3 or delta.shape != u.shape or u.shape[-1] == 0:
        raise ValueError("u and delta must have matching nonempty (batch, channels, length) shapes")
    batch, channels, length = u.shape
    if A.ndim != 2 or A.shape[0] != channels:
        raise ValueError("A must have shape (channels, state)")
    states = A.shape[1]
    original_dtype = u.dtype
    # Keep float64 for numerical checks; mixed/half precision accumulates in float32.
    dtype = torch.float64 if u.dtype == torch.float64 else torch.float32
    u, delta, A, B, C = [x.to(dtype) for x in (u, delta, A, B, C)]
    if delta_bias is not None:
        delta = delta + delta_bias.to(dtype)[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)

    def term_at(term, index):
        if term.ndim == 2 and term.shape == (channels, states):
            return term.unsqueeze(0)
        if term.ndim == 3 and term.shape == (batch, states, length):
            return term[:, None, :, index]
        if (term.ndim == 4 and term.shape[0] == batch and
                term.shape[2:] == (states, length) and channels % term.shape[1] == 0):
            return term[:, :, :, index].repeat_interleave(channels // term.shape[1], dim=1)
        raise ValueError("Invalid B/C state or group dimensions")

    state = u.new_zeros(batch, channels, states)
    outputs = []
    for index in range(length):
        step = delta[:, :, index, None]
        state = torch.exp(step * A) * state + step * term_at(B, index) * u[:, :, index, None]
        outputs.append((state * term_at(C, index)).sum(dim=-1))
    output = torch.stack(outputs, dim=-1)
    if D is not None:
        output = output + u * D.to(dtype)[None, :, None]
    if z is not None:
        output = output * F.silu(z.to(dtype))
    output = output.to(original_dtype)
    # Match the accelerated API: last state is diagnostic, not a gradient output.
    return (output, state.detach()) if return_last_state else output
