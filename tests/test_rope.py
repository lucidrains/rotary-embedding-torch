from __future__ import annotations

import pytest
import torch

from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb


def manual_apply_rope(input_vec: torch.Tensor, position: int | float, dim: int, theta: float) -> torch.Tensor:
    """Manually applies rotary positional embedding to the input vector.

    This function computes the rotary frequencies based on the given theta and dimension, and calculates the
    cosine and sine values for the position, and applies the rotation to pairs of elements in the input vector.

    Args:
        input_vec: The input tensor to apply rotations to. The last dimension should match `dim`.
        position: The position index for which to compute the embedding.
        dim: The embedding dimension. Must be even.
        theta: The base value for computing frequencies (e.g., 10000).

    Returns:
        The input vector with rotary embeddings applied.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    pos_freqs = position * freqs
    cos_vals = torch.cos(pos_freqs)
    sin_vals = torch.sin(pos_freqs)
    output = input_vec.clone()
    for i in range(dim // 2):
        j = 2 * i
        x0 = input_vec[..., j]
        x1 = input_vec[..., j + 1]
        output[..., j] = x0 * cos_vals[i] - x1 * sin_vals[i]
        output[..., j + 1] = x0 * sin_vals[i] + x1 * cos_vals[i]
    return output


def compute_expected_freqs(dim: int, theta: float) -> torch.Tensor:
    """Computes the expected base frequencies for RoPE.

    Args:
        dim: The embedding dimension.
        theta: The base value for computing frequencies.

    Returns:
        The expected base frequencies.
    """
    return 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))


@pytest.mark.parametrize("dim", [4, 8, 16, 32])
@pytest.mark.parametrize("theta", [10, 1000, 10000, 100000])
class TestSimpleRopeApply:
    @pytest.mark.parametrize("pos", [0, 1, 42, 13, 67, 69, 39])
    def test_apply_rotation(self, dim: int, theta: float, pos: int) -> None:
        """Test RoPE implementation against manually computed values for multiple examples."""
        # Generate dynamic input_vec using a seed based on pos for reproducibility
        torch.manual_seed(pos)
        input_vec = torch.randn(1, dim)

        # Create RoPE instance without caching to avoid issues
        rope = RotaryEmbedding(dim=dim, theta=theta, cache_if_possible=False)

        # Compute using library
        freqs = rope.forward(torch.tensor([pos]), seq_len=1)
        result = apply_rotary_emb(freqs, input_vec)

        # Compute expected using manual function
        expected = manual_apply_rope(input_vec, pos, dim, theta)

        # Assert close
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_frequencies(self, dim: int, theta: float) -> None:
        """Test that base frequencies are calculated correctly."""
        rope = RotaryEmbedding(dim=dim, theta=theta)

        actual_freqs = rope.freqs
        expected_freqs = compute_expected_freqs(dim, theta)
        torch.testing.assert_close(actual_freqs, expected_freqs, atol=1e-6, rtol=1e-6)
