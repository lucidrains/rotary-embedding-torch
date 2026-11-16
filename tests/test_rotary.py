import pytest
import torch

from rotary_embedding_torch import (
    get_yarn_freqs,
    RotaryEmbedding,
    apply_rotary_emb
)

from rotary_embedding_torch.flash_attn_with_rotary import get_flash_attention_fused

# helpers

def assert_allclose(a, b, atol = 1e-3):
    assert torch.allclose(a, b, atol = atol)

# tests

def test_basic_usage():
    rotary_emb = RotaryEmbedding(dim = 32)
    q = torch.randn(1, 8, 1024, 64)
    k = torch.randn(1, 8, 1024, 64)

    q_out = rotary_emb.rotate_queries_or_keys(q)
    k_out = rotary_emb.rotate_queries_or_keys(k)

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape

def test_yarn_frequency_bands():
    # With these bounds the transition spans bands 1 through 3.
    positions = torch.tensor([0, 1, 8193])
    angles, scale = get_yarn_freqs(positions, 8, original_max_seq_len = 25, factor = 4,
                                    theta = 16, beta_fast = 1.5, beta_slow = 0.75)
    expected = torch.tensor([1., 0.5, 0.25 * 0.625, 0.125 / 4]).repeat_interleave(2)
    torch.testing.assert_close(angles, positions[:, None] * expected)
    assert scale == pytest.approx(1.1386294361)

def test_yarn_factor_one():
    positions = torch.arange(32)
    freqs, scale = get_yarn_freqs(positions, 64, original_max_seq_len = 2048, factor = 1)
    torch.testing.assert_close(freqs, RotaryEmbedding(64)(positions))
    assert scale == 1.

def test_yarn_cached_query_matches_full_attention():
    torch.manual_seed(0)
    q, k, v = torch.randn(3, 1, 2, 17, 48)
    positions = torch.arange(4096, 4113)
    angles, scale = get_yarn_freqs(positions, 32, original_max_seq_len = 2048, factor = 4)
    q_full = apply_rotary_emb(angles, q, scale = scale)
    k = apply_rotary_emb(angles, k, scale = scale)
    last, scale = get_yarn_freqs(positions[-1:], 32, original_max_seq_len = 2048, factor = 4)
    q_last = apply_rotary_emb(last, q[..., -1:, :], scale = scale)
    attend = torch.nn.functional.scaled_dot_product_attention
    torch.testing.assert_close(attend(q_last, k, v), attend(q_full, k, v)[..., -1:, :])
    torch.testing.assert_close(q_full[..., 32:], q[..., 32:])

def test_inverse_rotation():
    rotary_emb = RotaryEmbedding(dim = 32)
    t = torch.randn(1, 8, 1024, 64)

    rotated = rotary_emb.rotate_queries_or_keys(t)
    derotated = rotary_emb.rotate_queries_or_keys(rotated, inverse = True)

    assert_allclose(derotated, t)

    # functional api, partial rotation

    freqs = rotary_emb(torch.arange(t.shape[-2]))

    rotated = apply_rotary_emb(freqs, t, start_index = 16)
    derotated = apply_rotary_emb(freqs, rotated, start_index = 16, inverse = True)

    assert_allclose(derotated, t)

def test_inference_key_value_cache():
    rotary_emb = RotaryEmbedding(dim = 32)
    q = torch.randn(1, 8, 1, 64)
    k = torch.randn(1, 8, 1024, 64)

    q_out, k_out = rotary_emb.rotate_queries_with_cached_keys(q, k)

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape

def test_axial_rotary_embeddings():
    pos_emb = RotaryEmbedding(
        dim = 16,
        freqs_for = 'pixel',
        max_freq = 256
    )

    q = torch.randn(1, 8, 64, 32, 64)
    k = torch.randn(1, 8, 64, 32, 64)

    freqs = pos_emb.get_axial_freqs(8, 64, 32)

    q_out = apply_rotary_emb(freqs, q)
    k_out = apply_rotary_emb(freqs, k)

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape

def test_length_extrapolatable_rotary_embeddings():
    rotary_emb = RotaryEmbedding(
        dim = 32,
        use_xpos = True
    )

    q = torch.randn(1, 8, 1024, 64)
    k = torch.randn(1, 8, 1024, 64)

    q_out, k_out = rotary_emb.rotate_queries_and_keys(q, k)

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape

def test_interpolating_sequence_positions():
    rotary_emb = RotaryEmbedding(
        dim = 32,
        interpolate_factor = 2.
    )

    freqs = rotary_emb(torch.arange(1024))
    assert freqs.shape[0] == 1024

# fused flash attention tests

@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'CUDA not available')
@pytest.mark.parametrize('seq_len', [128, 256, 1024])
@pytest.mark.parametrize('is_causal', [True, False])
@pytest.mark.parametrize('use_attn_mask', [True, False])
def test_flash_attn_with_rotary_matches_reference(seq_len, is_causal, use_attn_mask):
    flash_attn_triton = get_flash_attention_fused(force_reference = False)
    flash_attn_ref = get_flash_attention_fused(force_reference = True)

    rotary_emb = RotaryEmbedding(dim = 32)
    freqs = rotary_emb(torch.arange(seq_len)).cuda()

    q = torch.randn(1, 8, seq_len, 64).cuda()
    k = torch.randn(1, 8, seq_len, 64).cuda()
    v = torch.randn(1, 8, seq_len, 64).cuda()

    attn_mask = None
    if use_attn_mask:
        attn_mask = torch.ones(1, seq_len, dtype = torch.bool).cuda()
        # mask out the last quarter
        attn_mask[0, seq_len - (seq_len // 4):] = False

    out_triton = flash_attn_triton(
        q, k, v,
        rotary_pos_emb = freqs,
        attn_mask = attn_mask,
        is_causal = is_causal
    )
    
    out_ref = flash_attn_ref(
        q, k, v,
        rotary_pos_emb = freqs,
        attn_mask = attn_mask,
        is_causal = is_causal
    )

    assert_allclose(out_triton, out_ref)

@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'CUDA not available')
def test_mixed_position_readme():
    flash_attn_triton = get_flash_attention_fused(force_reference = False)
    flash_attn_ref = get_flash_attention_fused(force_reference = True)

    rotary_emb = RotaryEmbedding(dim = 32)
    freqs = rotary_emb(torch.arange(1024)).cuda()

    q = torch.randn(1, 8, 1026, 64).cuda()
    k = torch.randn(1, 8, 1026, 64).cuda()
    v = torch.randn(1, 8, 1026, 64).cuda()

    pos_indices = torch.arange(1024).cuda() + 2

    out_triton = flash_attn_triton(
        q, k, v,
        rotary_pos_emb = freqs,
        rotary_pos_emb_indices = pos_indices,
        is_causal = True
    )
    
    out_ref = flash_attn_ref(
        q, k, v,
        rotary_pos_emb = freqs,
        rotary_pos_emb_indices = pos_indices,
        is_causal = True
    )

    assert_allclose(out_triton, out_ref)

@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'CUDA not available')
def test_interspersed_register_tokens():
    flash_attn_triton = get_flash_attention_fused(force_reference = False)
    flash_attn_ref = get_flash_attention_fused(force_reference = True)

    seq_len = 256
    num_rotary_tokens = seq_len // 2

    rotary_emb = RotaryEmbedding(dim = 32)
    freqs = rotary_emb(torch.arange(num_rotary_tokens)).cuda()

    q = torch.randn(1, 8, seq_len, 64).cuda()
    k = torch.randn(1, 8, seq_len, 64).cuda()
    v = torch.randn(1, 8, seq_len, 64).cuda()

    # even tokens have rotary positions, odd tokens are register tokens
    pos_indices = torch.arange(0, seq_len, 2).cuda()

    out_triton = flash_attn_triton(
        q, k, v,
        rotary_pos_emb = freqs,
        rotary_pos_emb_indices = pos_indices,
        is_causal = True
    )
    
    out_ref = flash_attn_ref(
        q, k, v,
        rotary_pos_emb = freqs,
        rotary_pos_emb_indices = pos_indices,
        is_causal = True
    )

    assert_allclose(out_triton, out_ref)

@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'CUDA not available')
@pytest.mark.parametrize('seq_len', [128, 256])
@pytest.mark.parametrize('is_causal', [True, False])
def test_flash_attn_with_rotary_gqa(seq_len, is_causal):
    flash_attn_triton = get_flash_attention_fused(force_reference = False)
    flash_attn_ref = get_flash_attention_fused(force_reference = True)

    rotary_emb = RotaryEmbedding(dim = 32)
    freqs = rotary_emb(torch.arange(seq_len)).cuda()

    q_heads = 8
    kv_heads = 2

    q = torch.randn(1, q_heads, seq_len, 64).cuda()
    k = torch.randn(1, kv_heads, seq_len, 64).cuda()
    v = torch.randn(1, kv_heads, seq_len, 64).cuda()

    out_triton = flash_attn_triton(
        q, k, v,
        rotary_pos_emb = freqs,
        is_causal = is_causal
    )
    
    out_ref = flash_attn_ref(
        q, k, v,
        rotary_pos_emb = freqs,
        is_causal = is_causal
    )

    assert_allclose(out_triton, out_ref)
