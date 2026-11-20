import pytest
import torch

from backend.skinvision.models.selective_scan import selective_scan_torch


def test_scalar_recurrence_and_gradients():
    u = torch.tensor([[[1., 2., 3.]]], requires_grad=True)
    delta = torch.ones_like(u, requires_grad=True)
    a = torch.zeros(1, 1, requires_grad=True)
    b = torch.ones(1, 1, requires_grad=True)
    c = torch.ones(1, 1, requires_grad=True)
    out, last = selective_scan_torch(u, delta, a, b, c, return_last_state=True)
    torch.testing.assert_close(out, torch.tensor([[[1., 3., 6.]]]))
    torch.testing.assert_close(last, torch.tensor([[[6.]]]))
    out.sum().backward()
    torch.testing.assert_close(u.grad, torch.tensor([[[3., 2., 1.]]]))
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in [delta, a, b, c])


def test_grouped_terms_match_expanded_terms_and_gradients():
    torch.manual_seed(42)
    args = [torch.randn(2, 4, 7), torch.randn(2, 4, 7), -torch.rand(4, 3),
            torch.randn(2, 2, 3, 7), torch.randn(2, 2, 3, 7)]
    args = [x.requires_grad_() for x in args]
    kwargs = dict(D=torch.randn(4), z=torch.randn(2, 4, 7),
                  delta_bias=torch.randn(4), delta_softplus=True)
    actual = selective_scan_torch(*args, **kwargs)
    expanded = [*args[:3], args[3].repeat_interleave(2, dim=1), args[4].repeat_interleave(2, dim=1)]
    expected = selective_scan_torch(*expanded, **kwargs)
    torch.testing.assert_close(actual, expected)
    g1 = torch.autograd.grad(actual.sum(), args, retain_graph=True)
    g2 = torch.autograd.grad(expected.sum(), args)
    for left, right in zip(g1, g2):
        torch.testing.assert_close(left, right)


def test_rejects_unsupported_complex_states():
    with pytest.raises(ValueError, match='real'):
        selective_scan_torch(torch.ones(1, 1, 1), torch.ones(1, 1, 1),
                             torch.ones(1, 1, dtype=torch.complex64), torch.ones(1, 1), torch.ones(1, 1))


def test_decay_matches_closed_form_and_finite_difference_gradients():
    # Constant scalar dynamics: h_t = sum_{k=0}^{t} exp(-0.5*k).
    import math
    u = torch.ones(1, 1, 4, dtype=torch.double, requires_grad=True)
    delta = torch.full_like(u, 0.5, requires_grad=True)
    a = torch.tensor([[-1.]], dtype=torch.double, requires_grad=True)
    b = torch.tensor([[2.]], dtype=torch.double, requires_grad=True)
    c = torch.tensor([[1.]], dtype=torch.double, requires_grad=True)
    expected = torch.tensor([[[sum(math.exp(-0.5*k) for k in range(t+1)) for t in range(4)]]], dtype=torch.double)
    torch.testing.assert_close(selective_scan_torch(u, delta, a, b, c), expected)
    assert torch.autograd.gradcheck(selective_scan_torch, (u, delta, a, b, c))
