import pytest
import torch

from rotary_embedding_torch import (
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

@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_long_sequence_positions_exact_in_low_precision(dtype):
    # get_seq_pos built the position arange in the query dtype; float16 /
    # bfloat16 cannot represent every integer past 2048 / 256, so distinct
    # positions on a long sequence collapsed onto the same rotary frequency.
    rotary_emb = RotaryEmbedding(dim = 64, cache_if_possible = False)

    seq_len = 4096
    q = torch.randn(1, 4, seq_len, 64)

    out_fp32 = rotary_emb.rotate_queries_or_keys(q)
    out_low = rotary_emb.rotate_queries_or_keys(q.to(dtype))

    tol = 1e-2 if dtype is torch.float16 else 6e-2
    assert torch.allclose(out_fp32, out_low.float(), atol = tol)

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
