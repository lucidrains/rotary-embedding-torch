import math
import torch
from torch import cat
from torch import Tensor
from einops import rearrange, repeat

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None

from rotary_embedding_torch.rotary_embedding_torch import rotate_half

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# reference flash attention

def reference_flash_attention(
    q, k, v,
    cos = None,
    sin = None,
    pos_mask = None,
    attn_mask = None,
    is_causal = True
):
    device, dtype = q.device, q.dtype
    batch, heads_q, seq_len_q, dim = q.shape
    _, heads_kv, seq_len_k, _ = k.shape

    if heads_q != heads_kv:
        groups = heads_q // heads_kv
        k = repeat(k, 'b h n d -> b (h g) n d', g = groups)
        v = repeat(v, 'b h n d -> b (h g) n d', g = groups)

    # attention similarity

    sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)

    # rotary embeddings

    if exists(cos) and exists(sin):
        rot_dim = cos.shape[-1]

        q_rot = (q[..., :rot_dim] * cos) + (rotate_half(q[..., :rot_dim]) * sin)
        k_rot = (k[..., :rot_dim] * cos) + (rotate_half(k[..., :rot_dim]) * sin)

        q_rot = cat((q_rot, q[..., rot_dim:]), dim = -1)
        k_rot = cat((k_rot, k[..., rot_dim:]), dim = -1)

        sim_rot = torch.einsum('b h i d, b h j d -> b h i j', q_rot, k_rot)

        # apply positional mask if provided

        if exists(pos_mask):
            is_pos_2d = torch.logical_and(pos_mask[:, None], pos_mask[None, :])
            sim = torch.where(is_pos_2d, sim_rot, sim)
        else:
            sim = sim_rot

    sim = sim * (dim ** -0.5)

    # masks

    if exists(attn_mask):
        sim = sim + attn_mask

    if is_causal:
        causal_mask = torch.ones((seq_len_q, seq_len_k), device = device, dtype = torch.bool).tril()
        sim = torch.where(causal_mask, sim, float('-inf'))

    # aggregate values

    fully_masked = (sim == float('-inf')).all(dim = -1, keepdim = True)
    attn = sim.masked_fill(fully_masked, 0.).softmax(dim = -1)
    attn = attn.masked_fill(fully_masked, 0.)

    return torch.einsum('b h i j, b h j d -> b h i d', attn, v)

def _dummy_kernel(*args, **kwargs): pass

# triton forward and backward kernels

if TRITON_AVAILABLE:
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=1, num_warps=4),
        ],
        key=['N_CTX_Q', 'N_CTX_K', 'BLOCK_DMODEL']
    )
    @triton.jit
    def _flash_rotary_fwd_kernel(
        Q, K, V, sm_scale,
        Cos, Sin,
        Out, Lse,
        PosMask, AttnMask,
        stride_ab, stride_ah, stride_am, stride_an,
        stride_qm, stride_qh, stride_qn, stride_qk,
        stride_km, stride_kh, stride_kn, stride_kk,
        stride_vm, stride_vh, stride_vn, stride_vk,
        stride_cz, stride_ch, stride_cn, stride_ck,
        stride_om, stride_oh, stride_on, stride_ok,
        batch, q_heads, kv_heads, N_CTX_Q, N_CTX_K,
        BLOCK_DMODEL: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        HAS_POS_MASK: tl.constexpr,
        HAS_ATTN_MASK: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)

        off_z = off_hz // q_heads
        off_h = off_hz % q_heads
        off_h_kv = off_h // (q_heads // kv_heads)

        q_offset = off_z * stride_qm + off_h * stride_qh
        k_offset = off_z * stride_km + off_h_kv * stride_kh
        v_offset = off_z * stride_vm + off_h_kv * stride_vh
        c_offset = off_z * stride_cz + off_h * stride_ch

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_DMODEL)

        q_ptrs = Q + q_offset + offs_m[:, None] * stride_qn + offs_k[None, :] * stride_qk
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < N_CTX_Q) & (offs_k[None, :] < BLOCK_DMODEL), other=0.0)

        offs_k_swapped = offs_k ^ 1
        rot_sign = ((offs_k % 2) * 2 - 1)

        q_swapped_ptrs = Q + q_offset + offs_m[:, None] * stride_qn + offs_k_swapped[None, :] * stride_qk
        q_swapped = tl.load(q_swapped_ptrs, mask=(offs_m[:, None] < N_CTX_Q) & (offs_k[None, :] < BLOCK_DMODEL), other=0.0)

        cos_q_ptrs = Cos + c_offset + offs_m[:, None] * stride_cn + offs_k[None, :] * stride_ck
        sin_q_ptrs = Sin + c_offset + offs_m[:, None] * stride_cn + offs_k[None, :] * stride_ck
        cos_q = tl.load(cos_q_ptrs, mask=(offs_m[:, None] < N_CTX_Q) & (offs_k[None, :] < ROTARY_DIM), other=1.0)
        sin_q = tl.load(sin_q_ptrs, mask=(offs_m[:, None] < N_CTX_Q) & (offs_k[None, :] < ROTARY_DIM), other=0.0)

        q_rot = q * cos_q + q_swapped * rot_sign[None, :] * sin_q

        if HAS_POS_MASK:
            pos_m_ptrs = PosMask + offs_m
            pos_m = tl.load(pos_m_ptrs, mask=offs_m < N_CTX_Q, other=0).to(tl.int1)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        for start_n in range(0, N_CTX_K, BLOCK_N):
            offs_n_curr = start_n + offs_n

            k_ptrs = K + k_offset + offs_n_curr[None, :] * stride_kn + offs_k[:, None] * stride_kk
            k = tl.load(k_ptrs, mask=(offs_n_curr[None, :] < N_CTX_K) & (offs_k[:, None] < BLOCK_DMODEL), other=0.0)

            k_swapped_ptrs = K + k_offset + offs_n_curr[None, :] * stride_kn + offs_k_swapped[:, None] * stride_kk
            k_swapped = tl.load(k_swapped_ptrs, mask=(offs_n_curr[None, :] < N_CTX_K) & (offs_k[:, None] < BLOCK_DMODEL), other=0.0)

            cos_k_ptrs = Cos + c_offset + offs_n_curr[None, :] * stride_cn + offs_k[:, None] * stride_ck
            sin_k_ptrs = Sin + c_offset + offs_n_curr[None, :] * stride_cn + offs_k[:, None] * stride_ck
            cos_k = tl.load(cos_k_ptrs, mask=(offs_n_curr[None, :] < N_CTX_K) & (offs_k[:, None] < ROTARY_DIM), other=1.0)
            sin_k = tl.load(sin_k_ptrs, mask=(offs_n_curr[None, :] < N_CTX_K) & (offs_k[:, None] < ROTARY_DIM), other=0.0)

            k_rot = k * cos_k + k_swapped * rot_sign[:, None] * sin_k

            sim_rot = tl.dot(q_rot, k_rot, allow_tf32=False) * sm_scale

            if HAS_POS_MASK:
                sim_unrot = tl.dot(q, k, allow_tf32=False) * sm_scale
                pos_k_ptrs = PosMask + offs_n_curr
                pos_k = tl.load(pos_k_ptrs, mask=offs_n_curr < N_CTX_K, other=0).to(tl.int1)
                is_both = pos_m[:, None] & pos_k[None, :]
                sim = tl.where(is_both, sim_rot, sim_unrot)
            else:
                sim = sim_rot

            if HAS_ATTN_MASK:
                attn_mask_ptrs = AttnMask + off_z * stride_ab + off_h * stride_ah + offs_m[:, None] * stride_am + offs_n_curr[None, :] * stride_an
                attn_mask_val = tl.load(attn_mask_ptrs, mask=(offs_m[:, None] < N_CTX_Q) & (offs_n_curr[None, :] < N_CTX_K), other=float("-inf"))
                sim += attn_mask_val

            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
                sim = tl.where(causal_mask, sim, float("-inf"))

            sim = tl.where(offs_n_curr[None, :] < N_CTX_K, sim, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(sim, 1))
            p = tl.math.exp(sim - m_ij[:, None])
            p = tl.where(sim == float("-inf"), 0.0, p)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp(m_i - m_ij)
            alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]

            v_ptrs = V + v_offset + offs_n_curr[:, None] * stride_vn + offs_k[None, :] * stride_vk
            v = tl.load(v_ptrs, mask=(offs_n_curr[:, None] < N_CTX_K) & (offs_k[None, :] < BLOCK_DMODEL), other=0.0)
            acc += tl.dot(p.to(V.dtype.element_ty), v, allow_tf32=False)
            m_i = m_ij

        l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
        acc = acc / l_i_safe[:, None]

        # write back lse
        lse_ptrs = Lse + off_hz * N_CTX_Q + offs_m
        tl.store(lse_ptrs, m_i + tl.math.log(l_i_safe), mask=offs_m < N_CTX_Q)

        out_offset = off_z * stride_om + off_h * stride_oh
        out_ptrs = Out + out_offset + offs_m[:, None] * stride_on + offs_k[None, :] * stride_ok
        tl.store(out_ptrs, acc.to(Q.dtype.element_ty), mask=(offs_m[:, None] < N_CTX_Q) & (offs_k[None, :] < BLOCK_DMODEL))

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=1, num_warps=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=1, num_warps=4),
        ],
        key=['N_CTX_Q', 'N_CTX_K', 'BLOCK_DMODEL'],
        reset_to_zero=['DQ', 'DK', 'DV']
    )
    @triton.jit
    def _flash_rotary_bwd_kernel(
        Q, K, V, sm_scale,
        Cos, Sin,
        Out, DO, Lse,
        DQ, DK, DV,
        PosMask, AttnMask,
        stride_ab, stride_ah, stride_am, stride_an,
        stride_qm, stride_qh, stride_qn, stride_qk,
        stride_km, stride_kh, stride_kn, stride_kk,
        stride_vm, stride_vh, stride_vn, stride_vk,
        stride_cz, stride_ch, stride_cn, stride_ck,
        stride_om, stride_oh, stride_on, stride_ok,
        stride_dom, stride_doh, stride_don, stride_dok,
        batch, q_heads, kv_heads, N_CTX_Q, N_CTX_K,
        BLOCK_DMODEL: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        HAS_POS_MASK: tl.constexpr,
        HAS_ATTN_MASK: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr
    ):
        start_n = tl.program_id(0)
        off_hz = tl.program_id(1)

        off_z = off_hz // q_heads
        off_h = off_hz % q_heads
        off_h_kv = off_h // (q_heads // kv_heads)

        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_DMODEL)

        q_offset = off_z * stride_qm + off_h * stride_qh
        k_offset = off_z * stride_km + off_h_kv * stride_kh
        v_offset = off_z * stride_vm + off_h_kv * stride_vh
        o_offset = off_z * stride_om + off_h * stride_oh
        c_offset = off_z * stride_cz + off_h * stride_ch

        k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
        v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
        dk_ptrs = DK + k_offset + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
        dv_ptrs = DV + v_offset + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk

        k = tl.load(k_ptrs, mask=(offs_n[:, None] < N_CTX_K) & (offs_k[None, :] < BLOCK_DMODEL), other=0.0)
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < N_CTX_K) & (offs_k[None, :] < BLOCK_DMODEL), other=0.0)

        offs_k_swapped = offs_k ^ 1
        rot_sign = ((offs_k % 2) * 2 - 1)

        k_swapped_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_k_swapped[None, :] * stride_kk
        k_swapped = tl.load(k_swapped_ptrs, mask=(offs_n[:, None] < N_CTX_K) & (offs_k[None, :] < BLOCK_DMODEL), other=0.0)

        cos_k_ptrs = Cos + c_offset + offs_n[:, None] * stride_cn + offs_k[None, :] * stride_ck
        sin_k_ptrs = Sin + c_offset + offs_n[:, None] * stride_cn + offs_k[None, :] * stride_ck

        cos_k = tl.load(cos_k_ptrs, mask=(offs_n[:, None] < N_CTX_K) & (offs_k[None, :] < ROTARY_DIM), other=1.0)
        sin_k = tl.load(sin_k_ptrs, mask=(offs_n[:, None] < N_CTX_K) & (offs_k[None, :] < ROTARY_DIM), other=0.0)

        k_rot = k * cos_k + k_swapped * rot_sign[None, :] * sin_k
        k_rot_swapped = k_swapped * cos_k - k * rot_sign[None, :] * sin_k

        dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dk_rot = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dk_rot_swapped = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        if HAS_POS_MASK:
            dk_unrot = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

        if IS_CAUSAL:
            start_m = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M
        else:
            start_m = 0

        do_offset = off_z * stride_dom + off_h * stride_doh
        offs_m_init = start_m + tl.arange(0, BLOCK_M)
        q_ptrs = Q + q_offset + offs_m_init[:, None] * stride_qn + offs_k[None, :] * stride_qk
        do_ptrs = DO + do_offset + offs_m_init[:, None] * stride_don + offs_k[None, :] * stride_dok
        out_ptrs = Out + o_offset + offs_m_init[:, None] * stride_on + offs_k[None, :] * stride_ok
        dq_ptrs = DQ + q_offset + offs_m_init[:, None] * stride_qn + offs_k[None, :] * stride_qk
        lse_ptrs = Lse + off_hz * N_CTX_Q + offs_m_init

        if HAS_POS_MASK:
            pos_k_ptrs = PosMask + offs_n
            pos_k = tl.load(pos_k_ptrs, mask=offs_n < N_CTX_K, other=0).to(tl.int1)

        for m in range(start_m, N_CTX_Q, BLOCK_M):
            offs_m = m + tl.arange(0, BLOCK_M)

            q = tl.load(q_ptrs, mask=(offs_m[:, None] < N_CTX_Q) & (offs_k[None, :] < BLOCK_DMODEL), other=0.0)
            q_swapped_ptrs = Q + q_offset + offs_m[:, None] * stride_qn + offs_k_swapped[None, :] * stride_qk
            q_swapped = tl.load(q_swapped_ptrs, mask=(offs_m[:, None] < N_CTX_Q) & (offs_k[None, :] < BLOCK_DMODEL), other=0.0)

            cos_q_ptrs = Cos + c_offset + offs_m[:, None] * stride_cn + offs_k[None, :] * stride_ck
            sin_q_ptrs = Sin + c_offset + offs_m[:, None] * stride_cn + offs_k[None, :] * stride_ck
            cos_q = tl.load(cos_q_ptrs, mask=(offs_m[:, None] < N_CTX_Q) & (offs_k[None, :] < ROTARY_DIM), other=1.0)
            sin_q = tl.load(sin_q_ptrs, mask=(offs_m[:, None] < N_CTX_Q) & (offs_k[None, :] < ROTARY_DIM), other=0.0)

            q_rot = q * cos_q + q_swapped * rot_sign[None, :] * sin_q
            q_rot_swapped = q_swapped * cos_q - q * rot_sign[None, :] * sin_q

            do = tl.load(do_ptrs, mask=(offs_m[:, None] < N_CTX_Q) & (offs_k[None, :] < BLOCK_DMODEL), other=0.0)
            out = tl.load(out_ptrs, mask=(offs_m[:, None] < N_CTX_Q) & (offs_k[None, :] < BLOCK_DMODEL), other=0.0)
            lse = tl.load(lse_ptrs, mask=offs_m < N_CTX_Q, other=0.0)

            sim_rot = tl.dot(q_rot, tl.trans(k_rot), allow_tf32=False) * sm_scale

            if HAS_POS_MASK:
                sim_unrot = tl.dot(q, tl.trans(k), allow_tf32=False) * sm_scale
                pos_m_ptrs = PosMask + offs_m
                pos_m = tl.load(pos_m_ptrs, mask=offs_m < N_CTX_Q, other=0).to(tl.int1)
                is_both = pos_m[:, None] & pos_k[None, :]
                sim = tl.where(is_both, sim_rot, sim_unrot)
            else:
                sim = sim_rot

            if HAS_ATTN_MASK:
                attn_mask_ptrs = AttnMask + off_z * stride_ab + off_h * stride_ah + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
                attn_mask_val = tl.load(attn_mask_ptrs, mask=(offs_m[:, None] < N_CTX_Q) & (offs_n[None, :] < N_CTX_K), other=float("-inf"))
                sim += attn_mask_val

            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= offs_n[None, :]
                sim = tl.where(causal_mask, sim, float("-inf"))

            sim = tl.where(offs_n[None, :] < N_CTX_K, sim, float("-inf"))
            sim = tl.where(offs_m[:, None] < N_CTX_Q, sim, float("-inf"))

            p = tl.math.exp(sim - lse[:, None])
            p = tl.where(sim == float("-inf"), 0.0, p)

            dp = tl.dot(do, tl.trans(v), allow_tf32=False)
            Di = tl.sum(do * out, axis=1)
            ds = p * (dp - Di[:, None])
            ds = tl.where(sim == float("-inf"), 0.0, ds)

            dv += tl.dot(tl.trans(p.to(V.dtype.element_ty)), do, allow_tf32=False)

            if HAS_POS_MASK:
                is_both_f32 = tl.where(is_both, 1.0, 0.0)
                ds_rot = ds * is_both_f32
                ds_unrot = ds * (1.0 - is_both_f32)

                dk_rot += tl.dot(tl.trans(ds_rot.to(Q.dtype.element_ty)), q_rot, allow_tf32=False) * sm_scale
                dk_rot_swapped += tl.dot(tl.trans(ds_rot.to(Q.dtype.element_ty)), q_rot_swapped, allow_tf32=False) * sm_scale
                dk_unrot += tl.dot(tl.trans(ds_unrot.to(Q.dtype.element_ty)), q, allow_tf32=False) * sm_scale

                dq_rot_chunk = tl.dot(ds_rot.to(K.dtype.element_ty), k_rot, allow_tf32=False) * sm_scale
                dq_rot_chunk_swapped = tl.dot(ds_rot.to(K.dtype.element_ty), k_rot_swapped, allow_tf32=False) * sm_scale
                dq_chunk = dq_rot_chunk * cos_q - dq_rot_chunk_swapped * rot_sign[None, :] * sin_q

                dq_unrot_chunk = tl.dot(ds_unrot.to(K.dtype.element_ty), k, allow_tf32=False) * sm_scale
                dq_chunk += dq_unrot_chunk
            else:
                dk_rot += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q_rot, allow_tf32=False) * sm_scale
                dk_rot_swapped += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q_rot_swapped, allow_tf32=False) * sm_scale

                dq_rot_chunk = tl.dot(ds.to(K.dtype.element_ty), k_rot, allow_tf32=False) * sm_scale
                dq_rot_chunk_swapped = tl.dot(ds.to(K.dtype.element_ty), k_rot_swapped, allow_tf32=False) * sm_scale
                dq_chunk = dq_rot_chunk * cos_q - dq_rot_chunk_swapped * rot_sign[None, :] * sin_q

            tl.atomic_add(dq_ptrs, dq_chunk.to(DQ.dtype.element_ty), mask=(offs_m[:, None] < N_CTX_Q) & (offs_k[None, :] < BLOCK_DMODEL))

            q_ptrs += BLOCK_M * stride_qn
            do_ptrs += BLOCK_M * stride_don
            out_ptrs += BLOCK_M * stride_on
            dq_ptrs += BLOCK_M * stride_qn
            lse_ptrs += BLOCK_M

        dk = dk_rot * cos_k - dk_rot_swapped * rot_sign[None, :] * sin_k
        if HAS_POS_MASK:
            dk += dk_unrot

        tl.atomic_add(dk_ptrs, dk.to(Q.dtype.element_ty), mask=(offs_n[:, None] < N_CTX_K) & (offs_k[None, :] < BLOCK_DMODEL))
        tl.atomic_add(dv_ptrs, dv.to(Q.dtype.element_ty), mask=(offs_n[:, None] < N_CTX_K) & (offs_k[None, :] < BLOCK_DMODEL))
else:
    _flash_rotary_fwd_kernel = _dummy_kernel
    _flash_rotary_bwd_kernel = _dummy_kernel

# fused attention autograd function

class FlashAttentionFused(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q, k, v,
        cos, sin,
        pos_mask = None,
        attn_mask = None,
        is_causal = True
    ):
        device, dtype = q.device, q.dtype
        batch, heads_q, seq_len_q, dim = q.shape
        _, heads_kv, seq_len_k, _ = k.shape

        sm_scale = dim ** -0.5

        # contiguous

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        cos = cos.contiguous()
        sin = sin.contiguous()

        # output buffers

        out = torch.empty_like(q)
        lse = torch.empty((batch, heads_q, seq_len_q), device = device, dtype = torch.float32)

        # triton kernel config

        grid = lambda META: (
            triton.cdiv(seq_len_q, META['BLOCK_M']),
            batch * heads_q,
            1
        )

        stride_cz = cos.stride(0) if cos.size(0) > 1 else 0
        stride_ch = cos.stride(1) if cos.size(1) > 1 else 0
        stride_cn = cos.stride(2)
        stride_ck = cos.stride(3)

        # optional masks

        has_pos_mask = exists(pos_mask)
        if has_pos_mask:
            pos_mask = pos_mask.contiguous()

        has_attn_mask = exists(attn_mask)
        stride_ab = stride_ah = stride_am = stride_an = 0

        if has_attn_mask:
            attn_mask = attn_mask.contiguous()
            stride_ab, stride_ah, stride_am, stride_an = attn_mask.stride()

        _flash_rotary_fwd_kernel[grid](
            q, k, v, sm_scale,
            cos, sin,
            out, lse,
            pos_mask, attn_mask,
            stride_ab, stride_ah, stride_am, stride_an,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            stride_cz, stride_ch, stride_cn, stride_ck,
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            batch, heads_q, heads_kv, seq_len_q, seq_len_k,
            BLOCK_DMODEL=dim,
            ROTARY_DIM=cos.shape[-1],
            IS_CAUSAL=is_causal,
            HAS_POS_MASK=has_pos_mask,
            HAS_ATTN_MASK=has_attn_mask
        )

        ctx.save_for_backward(q, k, v, out, lse, cos, sin, pos_mask, attn_mask)
        ctx.sm_scale = sm_scale
        ctx.is_causal = is_causal
        ctx.has_pos_mask = has_pos_mask
        ctx.has_attn_mask = has_attn_mask
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse, cos, sin, pos_mask, attn_mask = ctx.saved_tensors

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        batch, heads_q, seq_len_q, dim = q.shape
        _, heads_kv, seq_len_k, _ = k.shape

        grid = lambda META: (
            triton.cdiv(seq_len_k, META['BLOCK_N']),
            batch * heads_q,
            1
        )

        stride_cz = cos.stride(0) if cos.size(0) > 1 else 0
        stride_ch = cos.stride(1) if cos.size(1) > 1 else 0
        stride_cn = cos.stride(2)
        stride_ck = cos.stride(3)

        stride_ab = stride_ah = stride_am = stride_an = 0
        if ctx.has_attn_mask:
            stride_ab, stride_ah, stride_am, stride_an = attn_mask.stride()

        _flash_rotary_bwd_kernel[grid](
            q, k, v, ctx.sm_scale,
            cos, sin,
            out, dout, lse,
            dq, dk, dv,
            pos_mask, attn_mask,
            stride_ab, stride_ah, stride_am, stride_an,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            stride_cz, stride_ch, stride_cn, stride_ck,
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            batch, heads_q, heads_kv, seq_len_q, seq_len_k,
            BLOCK_DMODEL=dim,
            ROTARY_DIM=cos.shape[-1],
            IS_CAUSAL=ctx.is_causal,
            HAS_POS_MASK=ctx.has_pos_mask,
            HAS_ATTN_MASK=ctx.has_attn_mask
        )
        return dq, dk, dv, None, None, None, None, None

# factory

def get_flash_attention_fused(force_reference = False):
    def inner(
        q, k, v,
        cos = None,
        sin = None,
        pos_mask = None,
        attn_mask = None,
        is_causal = True,
        rotary_pos_emb = None,
        rotary_pos_emb_indices = None
    ):
        device, dtype = q.device, q.dtype
        batch, heads_q, seq_len_q, dim = q.shape
        _, heads_kv, seq_len_k, _ = k.shape

        # handle rotary positional embeddings natively

        if exists(rotary_pos_emb):
            rotary_pos_emb = rotary_pos_emb.to(device)

            if exists(rotary_pos_emb_indices):
                if rotary_pos_emb.ndim > 2:
                    rotary_pos_emb = rearrange(rotary_pos_emb, '... d -> (...) d')

                padded_freqs = torch.zeros((seq_len_q, rotary_pos_emb.shape[-1]), device = device, dtype = dtype)
                padded_freqs[rotary_pos_emb_indices] = rotary_pos_emb
                rotary_pos_emb = padded_freqs

                pos_mask = torch.zeros(seq_len_q, device = device, dtype = torch.bool)
                pos_mask[rotary_pos_emb_indices] = True

            cos = rearrange(rotary_pos_emb.cos(), '... n d -> ... 1 1 n d')
            sin = rearrange(rotary_pos_emb.sin(), '... n d -> ... 1 1 n d')

        assert exists(cos) and exists(sin), 'either cos/sin or rotary_pos_emb must be provided'

        # handle attention mask formatting

        if exists(attn_mask):
            if attn_mask.dtype == torch.bool:
                attn_mask_float = torch.zeros_like(attn_mask, dtype = dtype)
                attn_mask_float[~attn_mask] = -float('inf')
                attn_mask = attn_mask_float

            attn_mask = rearrange(attn_mask, 'b j -> b 1 1 j') if attn_mask.ndim == 2 else rearrange(attn_mask, 'b i j -> b 1 i j')
            attn_mask = attn_mask.expand(batch, heads_q, seq_len_q, seq_len_k)

        # dispatch to reference if requested or triton unavailable

        if force_reference or not TRITON_AVAILABLE:
            return reference_flash_attention(q, k, v, cos, sin, pos_mask, attn_mask, is_causal)

        return FlashAttentionFused.apply(q, k, v, cos, sin, pos_mask, attn_mask, is_causal)
    return inner

flash_attn_with_rotary = get_flash_attention_fused(force_reference = False)
