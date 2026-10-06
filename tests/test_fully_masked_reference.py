import pytest
import torch
import torch.nn.functional as F

from rotary_embedding_torch.flash_attn_with_rotary import get_flash_attention_fused


@pytest.mark.parametrize('is_causal', [False, True])
@pytest.mark.parametrize('mask_rank', [2, 3])
def test_fully_masked_reference_matches_sdpa_values_and_gradients(is_causal, mask_rank):
    torch.manual_seed(31)
    q = torch.randn(2, 4, 5, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 2, 5, 8, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 2, 5, 8, dtype=torch.float64, requires_grad=True)
    mask = torch.ones((2, 5) if mask_rank == 2 else (2, 5, 5), dtype=torch.bool)
    mask[0] = False
    if mask_rank == 3:
        mask[1, 2] = False
    else:
        mask[1, :2] = False
    expanded_mask = mask[:, None, None, :] if mask_rank == 2 else mask[:, None, :, :]
    if is_causal:
        expanded_mask = expanded_mask & torch.ones(5, 5, dtype=torch.bool).tril()
    expected = F.scaled_dot_product_attention(q, k.repeat_interleave(2, 1),
                                             v.repeat_interleave(2, 1), attn_mask=expanded_mask)
    actual = get_flash_attention_fused(force_reference=True)(
        q, k, v, rotary_pos_emb=torch.zeros(5, 4, dtype=torch.float64),
        attn_mask=mask, is_causal=is_causal)
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual[0], torch.zeros_like(actual[0]), rtol=0, atol=0)
    weights = torch.linspace(-0.8, 1.1, actual.numel(), dtype=actual.dtype).reshape(actual.shape)
    actual_grads = torch.autograd.grad((actual * weights).sum(), (q, k, v), retain_graph=True)
    expected_grads = torch.autograd.grad((expected * weights).sum(), (q, k, v))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, expected_grad)
