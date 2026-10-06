import pytest
import torch

from rotary_embedding_torch import RotaryEmbedding


def rotate_by_formula(t, frequencies, positions, scale=1.):
    angle = positions[:, None] * frequencies[None, :]
    even = t[..., :frequencies.numel() * 2:2]
    odd = t[..., 1:frequencies.numel() * 2:2]
    rotated = torch.stack((even * angle.cos() - odd * angle.sin(),
                           even * angle.sin() + odd * angle.cos()), dim=-1)
    rotated = rotated.flatten(-2) * scale
    return torch.cat((rotated, t[..., frequencies.numel() * 2:]), dim=-1)


@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('assign', [False, True])
def test_loading_new_frequencies_invalidates_warm_cache(nested, assign):
    rotary = RotaryEmbedding(dim=8).double()
    fresh = RotaryEmbedding(dim=8).double()
    holder = torch.nn.ModuleDict({'rotary': rotary}) if nested else rotary
    source = torch.nn.ModuleDict({'rotary': fresh}) if nested else fresh
    with torch.no_grad():
        fresh.freqs.copy_(torch.tensor([0.6, 0.17, 0.03, 0.008], dtype=torch.float64))
    q = torch.linspace(-1., 1., 48, dtype=torch.float64).reshape(1, 1, 6, 8).requires_grad_()
    rotary.rotate_queries_or_keys(q)
    holder.load_state_dict(source.state_dict(), assign=assign)
    actual = rotary.rotate_queries_or_keys(q)
    expected = rotate_by_formula(q, fresh.freqs, torch.arange(6).double())
    torch.testing.assert_close(actual, expected)
    weights = torch.linspace(0.1, 0.9, q.numel(), dtype=q.dtype).reshape(q.shape)
    actual_grad, = torch.autograd.grad((actual * weights).sum(), q, retain_graph=True)
    expected_grad, = torch.autograd.grad((expected * weights).sum(), q)
    torch.testing.assert_close(actual_grad, expected_grad)
    torch.testing.assert_close(rotary.rotate_queries_or_keys(q[..., :3, :]), expected[..., :3, :])
