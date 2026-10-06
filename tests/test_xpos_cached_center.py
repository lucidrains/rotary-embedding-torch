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


@pytest.mark.parametrize('q_len,k_len', [(1, 8), (2, 9), (3, 8), (5, 5)])
def test_cached_xpos_matches_full_prefix_formula_and_gradients(q_len, k_len):
    rotary = RotaryEmbedding(dim=8, use_xpos=True, xpos_scale_base=4,
                             cache_if_possible=False).double()
    q = torch.linspace(-1.2, 1.7, q_len * 8, dtype=torch.float64).reshape(1, 1, q_len, 8)
    k = torch.linspace(0.8, -0.9, k_len * 8, dtype=torch.float64).reshape(1, 1, k_len, 8)
    q.requires_grad_()
    k.requires_grad_()
    positions = torch.arange(k_len, dtype=torch.float64)
    scales = (rotary.scale[None, :] **
              ((positions - k_len // 2)[:, None] / rotary.scale_base)).repeat_interleave(2, -1)
    expected_q = rotate_by_formula(q, rotary.freqs, positions[-q_len:], scales[-q_len:])
    expected_k = rotate_by_formula(k, rotary.freqs, positions, scales.reciprocal())
    actual_q, actual_k = rotary.rotate_queries_with_cached_keys(q, k)
    expected_scores = expected_q @ expected_k.transpose(-1, -2)
    actual_scores = actual_q @ actual_k.transpose(-1, -2)
    torch.testing.assert_close(actual_scores, expected_scores)
    torch.testing.assert_close(actual_q, expected_q)
    torch.testing.assert_close(actual_k, expected_k)
    actual_grad = torch.autograd.grad(actual_scores.square().sum(), (q, k), retain_graph=True)
    expected_grad = torch.autograd.grad(expected_scores.square().sum(), (q, k))
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected)


def test_plain_rope_cached_positions_are_unchanged():
    rotary = RotaryEmbedding(dim=8, cache_if_possible=False).double()
    q = torch.ones(1, 2, 3, 10, dtype=torch.float64)
    k = torch.ones(1, 2, 7, 10, dtype=torch.float64)
    actual_q, actual_k = rotary.rotate_queries_with_cached_keys(q, k)
    torch.testing.assert_close(actual_q, rotate_by_formula(q, rotary.freqs, torch.arange(4, 7).double()))
    torch.testing.assert_close(actual_k, rotate_by_formula(k, rotary.freqs, torch.arange(7).double()))
