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


def axial_formula(rotary, dims, offsets):
    positions = torch.meshgrid(*(torch.arange(n, dtype=rotary.freqs.dtype)
                                for n in dims), indexing='ij')
    return torch.cat([((p + offsets[i])[..., None] * rotary.freqs).repeat_interleave(2, -1)
                      for i, p in enumerate(positions)], dim=-1)


@pytest.mark.parametrize('dims,offsets', [((3, 5), (2., -0.5)),
                                        ((5, 3), (-3., 4.25)),
                                        ((2, 4, 3), (0., 1.5, -2.))])
def test_axial_offsets_do_not_reuse_zero_origin_or_other_axis_cache(dims, offsets):
    rotary = RotaryEmbedding(dim=8).double()
    rotary.get_axial_freqs(9)
    offsets = torch.tensor(offsets, dtype=torch.float64)
    actual = rotary.get_axial_freqs(*dims, offsets=offsets)
    torch.testing.assert_close(actual, axial_formula(rotary, dims, offsets))
    second = offsets + 0.75
    torch.testing.assert_close(rotary.get_axial_freqs(*dims, offsets=second),
                               axial_formula(rotary, dims, second))
    torch.testing.assert_close(rotary.get_axial_freqs(*dims),
                               axial_formula(rotary, dims, torch.zeros(len(dims))))


def test_axial_offset_parameter_has_fresh_gradients_on_repeated_calls():
    rotary = RotaryEmbedding(dim=8).double()
    offsets = torch.nn.Parameter(torch.tensor([0.4, -0.7], dtype=torch.float64))
    for step in range(2):
        actual = rotary.get_axial_freqs(3, 4, offsets=offsets)
        expected = axial_formula(rotary, (3, 4), offsets)
        weights = torch.linspace(0.1, 1., actual.numel(), dtype=actual.dtype).reshape(actual.shape)
        actual_loss = (actual.sin() * weights).sum() + offsets.sum() * 0
        expected_loss = (expected.sin() * weights).sum()
        actual_grad, = torch.autograd.grad(actual_loss, offsets)
        expected_grad, = torch.autograd.grad(expected_loss, offsets)
        torch.testing.assert_close(actual_grad, expected_grad)
        with torch.no_grad():
            offsets.add_(0.2)
