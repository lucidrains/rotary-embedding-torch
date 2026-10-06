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


@pytest.mark.parametrize('configuration', [dict(dim=8, freqs_for='constant', num_freqs=1),
                                         dict(dim=10, freqs_for='constant', num_freqs=3),
                                         dict(dim=8, custom_freqs=torch.tensor([0.7, 0.12, 0.03])),
                                         dict(dim=9), dict(dim=8)])
def test_cached_frequency_width_matches_rotating_pairs(configuration):
    rotary = RotaryEmbedding(**configuration).double()
    q = torch.linspace(-0.9, 1.3, 60, dtype=torch.float64).reshape(1, 1, 5, 12)
    q.requires_grad_()
    expected = rotate_by_formula(q, rotary.freqs, torch.arange(5).double())
    for length in (5, 3, 5):
        actual = rotary.rotate_queries_or_keys(q[..., :length, :])
        torch.testing.assert_close(actual, expected[..., :length, :])
        end = rotary.freqs.numel() * 2
        torch.testing.assert_close(actual[..., end:], q[..., :length, end:], rtol=0, atol=0)
        actual_grad, = torch.autograd.grad(actual.square().sum(), q, retain_graph=True)
        expected_grad, = torch.autograd.grad(expected[..., :length, :].square().sum(), q,
                                             retain_graph=True)
        torch.testing.assert_close(actual_grad, expected_grad)
