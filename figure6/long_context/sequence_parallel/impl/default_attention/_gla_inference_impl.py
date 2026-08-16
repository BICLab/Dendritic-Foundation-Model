import torch
import triton
import triton.language as tl

from typing import Optional, Tuple
from ._gla_utils import input_guard, autocast_custom_fwd, autocast_custom_bwd

@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['BT']
)
@triton.jit(do_not_specialize=['T'])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    offsets,
    indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    REVERSE: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    else:
        p_s = tl.make_block_ptr(s + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    # [BT]
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BS': BS}, num_warps=num_warps)
        for BS in [16, 32, 64]
        for num_warps in [2, 4, 8]
    ],
    key=['S', 'BT']
)
@triton.jit(do_not_specialize=['T'])
def chunk_local_cumsum_vector_kernel(
    s,
    o,
    offsets,
    indices,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    REVERSE: tl.constexpr
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, BT)
    if REVERSE:
        m_s = tl.where(o_i[:, None] <= o_i[None, :], 1., 0.)
    else:
        m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + (bos * H + i_h*T)*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_o = tl.make_block_ptr(o + (bos * H + i_h*T)*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    else:
        p_s = tl.make_block_ptr(s + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_o = tl.make_block_ptr(o + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.dot(m_s, b_s, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BT': 16}, num_warps=2),
        triton.Config({'BT': 32}, num_warps=4),
        triton.Config({'BT': 32}, num_warps=2),
        triton.Config({'BT': 64}, num_warps=8),
        triton.Config({'BT': 64}, num_warps=4),
    ],
    key=[]
)
@triton.jit(do_not_specialize=['T'])
def chunk_global_cumsum_scalar_kernel(
    s,
    o,
    offsets,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    REVERSE: tl.constexpr
):
    i_bh = tl.program_id(0)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_b).to(tl.int32), tl.load(offsets + i_b + 1).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos

    b_z = tl.zeros([], dtype=tl.float32)
    NT = tl.cdiv(T, BT)
    for i_c in range(NT):
        i_t = NT-1-i_c if REVERSE else i_c
        if HEAD_FIRST:
            p_s = tl.make_block_ptr(s + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
            p_o = tl.make_block_ptr(o + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        else:
            p_s = tl.make_block_ptr(s + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            p_o = tl.make_block_ptr(o + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
        b_o = tl.cumsum(b_s, axis=0)
        b_ss = tl.sum(b_s, 0)
        if REVERSE:
            b_o = -b_o + b_ss + b_s
        b_o += b_z
        if i_c >= 0:
            b_z += b_ss
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps)
        for BT in [16, 32, 64]
        for num_warps in [2, 4, 8]
    ],
    key=['S']
)
@triton.jit(do_not_specialize=['T'])
def chunk_global_cumsum_vector_kernel(
    s,
    z,
    offsets,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    REVERSE: tl.constexpr
):
    i_s, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_b).to(tl.int32), tl.load(offsets + i_b + 1).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos

    o_i = tl.arange(0, BT)
    if REVERSE:
        m_s = tl.where(o_i[:, None] <= o_i[None, :], 1., 0.)
    else:
        m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)

    b_z = tl.zeros([BS], dtype=tl.float32)
    NT = tl.cdiv(T, BT)
    for i_c in range(NT):
        i_t = NT-1-i_c if REVERSE else i_c
        if HEAD_FIRST:
            p_s = tl.make_block_ptr(s + (bos * H + i_h*T)*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
            p_z = tl.make_block_ptr(z + (bos * H + i_h*T)*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        else:
            p_s = tl.make_block_ptr(s + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
            p_z = tl.make_block_ptr(z + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        # [BT, BS]
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=False)
        tl.store(p_z, b_c.to(p_z.dtype.element_ty), boundary_check=(0, 1))
        if i_c >= 0:
            b_z += tl.sum(b_s, 0)


def chunk_local_cumsum_scalar(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    offsets: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    head_first: bool = True,
    output_dtype: Optional[torch.dtype] = torch.float
) -> torch.Tensor:
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    if offsets is not None:
        B = len(offsets) - 1
    assert chunk_size == 2**(chunk_size.bit_length()-1), "chunk_size must be a power of 2"
    BT = chunk_size
    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = torch.cat([
                torch.stack([offsets.new_full((n,), i), offsets.new_tensor(range(n))], 1)
                for i, n in enumerate(triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist())
            ])
        NT = len(indices)
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    grid = (NT, B * H)
    chunk_local_cumsum_scalar_kernel[grid](
        g_org,
        g,
        offsets,
        indices,
        T=T,
        H=H,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse
    )
    return g


def chunk_local_cumsum_vector(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    offsets: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    head_first: bool = True,
    output_dtype: Optional[torch.dtype] = torch.float
) -> torch.Tensor:
    if head_first:
        B, H, T, S = g.shape
    else:
        B, T, H, S = g.shape
    BT = chunk_size
    assert chunk_size == 2**(chunk_size.bit_length()-1), "chunk_size must be a power of 2"
    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = torch.cat([
                torch.stack([offsets.new_full((n,), i), offsets.new_tensor(range(n))], 1)
                for i, n in enumerate(triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist())
            ])
        NT = len(indices)
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    def grid(meta): return (triton.cdiv(meta['S'], meta['BS']), NT, B * H)
    # keep cummulative normalizer in fp32
    # this kernel is equivalent to
    # g = g.view(B, H, NT, BT, -1).cumsum(-2).view(B, H, T, -1)
    chunk_local_cumsum_vector_kernel[grid](
        g_org,
        g,
        offsets,
        indices,
        T=T,
        H=H,
        S=S,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse
    )
    return g


@input_guard
def chunk_global_cumsum_scalar(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    reverse: bool = False,
    offsets: Optional[torch.Tensor] = None,
    head_first: bool = True,
    output_dtype: Optional[torch.dtype] = torch.float
) -> torch.Tensor:
    dtype = dtype or s.dtype
    if head_first:
        B, H, T = s.shape
    else:
        B, T, H = s.shape
    if offsets is not None:
        B = len(offsets) - 1
    grid = (B * H,)
    z = torch.empty_like(s, dtype=output_dtype or dtype)
    chunk_global_cumsum_scalar_kernel[grid](
        s,
        z,
        offsets,
        T=T,
        H=H,
        HEAD_FIRST=head_first,
        REVERSE=reverse
    )
    return z


@input_guard
def chunk_global_cumsum_vector(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    reverse: bool = False,
    offsets: Optional[torch.Tensor] = None,
    head_first: bool = True,
    output_dtype: Optional[torch.dtype] = torch.float
) -> torch.Tensor:
    dtype = dtype or s.dtype
    if head_first:
        B, H, T, S = s.shape
    else:
        B, T, H, S = s.shape
    BS = min(32, triton.next_power_of_2(S))
    if offsets is not None:
        B = len(offsets) - 1
    grid = (triton.cdiv(S, BS), B * H)
    z = torch.empty_like(s, dtype=output_dtype or dtype)
    chunk_global_cumsum_vector_kernel[grid](
        s,
        z,
        offsets,
        T=T,
        H=H,
        S=S,
        BS=BS,
        HEAD_FIRST=head_first,
        REVERSE=reverse
    )
    return z


@input_guard
def chunk_global_cumsum(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    reverse: bool = False,
    offsets: Optional[torch.Tensor] = None,
    head_first: bool = True,
    output_dtype: Optional[torch.dtype] = torch.float
) -> torch.Tensor:
    if offsets is not None:
        assert s.shape[0] == 1, "Only batch size 1 is supported when offsets are provided"
    if len(s.shape) == 3:
        return chunk_global_cumsum_scalar(s, dtype, reverse, offsets, head_first, output_dtype)
    elif len(s.shape) == 4:
        return chunk_global_cumsum_vector(s, dtype, reverse, offsets, head_first, output_dtype)
    else:
        raise ValueError(f"Unsupported input shape {s.shape}. "
                         f"which should be [B, H, T]/[B, H, T, D] if `head_first=True` "
                         f"or [B, T, H]/[B, T, H, D] otherwise")


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=["BK", "BV", "USE_GK", "USE_GV", "USE_G"],
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_fwd_kernel(
    q,
    k,
    v,
    g,
    gk,
    gv,
    o,
    h0,
    ht,
    offsets,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    # indices
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int64), tl.load(offsets + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if HEAD_FIRST:
        p_q = q + i_nh * T*K + ((T-1) * K if REVERSE else 0) + i_k * BK + tl.arange(0, BK)
        p_k = k + i_nh * T*K + ((T-1) * K if REVERSE else 0) + i_k * BK + tl.arange(0, BK)
        p_v = v + i_nh * T*V + ((T-1) * V if REVERSE else 0) + i_v * BV + tl.arange(0, BV)
        p_o = o + (i_k * B*H + i_nh) * T*V + ((T-1) * V if REVERSE else 0) + i_v * BV + tl.arange(0, BV)
        if USE_G:
            p_g = g + i_nh * T + ((T-1) if REVERSE else 0)
        if USE_GK:
            p_gk = gk + i_nh * T*K + ((T-1) * K if REVERSE else 0) + i_k * BK + tl.arange(0, BK)
        if USE_GV:
            p_gv = gv + i_nh * T*V + ((T-1) * V if REVERSE else 0) + i_v * BV + tl.arange(0, BV)
    else:
        p_q = q + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_k = k + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_v = v + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)
        p_o = o + ((i_k * all + bos) + ((T-1) if REVERSE else 0)) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)
        if USE_G:
            p_g = g + (bos + ((T-1) if REVERSE else 0)) * H + i_h
        if USE_GK:
            p_gk = gk + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        if USE_GV:
            p_gv = gv + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)

    mask_k = (i_k * BK + tl.arange(0, BK)) < K
    mask_v = (i_v * BV + tl.arange(0, BV)) < V
    mask_h = mask_k[None, :] & mask_v[:, None]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K*V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
            b_h = b_h * tl.exp(b_gk[None, :])
        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_v, other=0).to(tl.float32)
            b_h = b_h * tl.exp(b_gv[:, None])
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_h = b_h * tl.exp(b_g)
        b_h += b_k[None, :] * b_v[:, None]
        b_o = b_h * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        p_q += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        p_k += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        p_v += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * V
        p_o += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * V
        if USE_GK:
            p_gk += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        if USE_GV:
            p_gv += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * V
        if USE_G:
            p_g += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H)

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K*V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_INITIAL_STATE_GRADIENT': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BK', 'BV', 'USE_GK', 'USE_GV', 'USE_G'],
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_bwd_kernel(
    q,
    k,
    v,
    g,
    gk,
    gv,
    h0,
    do,
    dq,
    dk,
    dv,
    dht,
    dh0,
    offsets,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int64), tl.load(offsets + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if HEAD_FIRST:
        p_k = k + i_nh * T*K + ((T-1) * K if REVERSE else 0) + i_k * BK + tl.arange(0, BK)
        p_v = v + i_nh * T*V + ((T-1) * V if REVERSE else 0) + i_v * BV + tl.arange(0, BV)
        p_do = do + i_nh * T*V + ((T-1) * V if REVERSE else 0) + i_v * BV + tl.arange(0, BV)
        p_dq = dq + (i_v * B*H + i_nh) * T*K + ((T-1) * K if REVERSE else 0) + i_k * BK + tl.arange(0, BK)
        if USE_G:
            p_g = g + i_nh * T + ((T-1) if REVERSE else 0)
        if USE_GK:
            p_gk = gk + i_nh * T*K + ((T-1) * K if REVERSE else 0) + i_k * BK + tl.arange(0, BK)
        if USE_GV:
            p_gv = gv + i_nh * T*V + ((T-1) * V if REVERSE else 0) + i_v * BV + tl.arange(0, BV)
    else:
        p_k = k + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_v = v + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)
        p_do = do + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)
        p_dq = dq + ((i_v * all + bos) + ((T-1) if REVERSE else 0)) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        if USE_G:
            p_g = g + (bos + ((T-1) if REVERSE else 0)) * H + i_h
        if USE_GK:
            p_gk = gk + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        if USE_GV:
            p_gv = gv + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)

    mask_k = i_k * BK + tl.arange(0, BK) < K
    mask_v = i_v * BV + tl.arange(0, BV) < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K*V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_h = b_h * tl.exp(b_g)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
            b_h = b_h * tl.exp(b_gk[:, None])
        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_v, other=0).to(tl.float32)
            b_h = b_h * tl.exp(b_gv[None, :])
        b_h += b_k[:, None] * b_v[None, :]
        b_dq = b_h * b_do[None, :]
        b_dq = tl.sum(b_dq, axis=1) * scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=mask_k)

        p_k += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        p_v += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * V
        p_do += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * V
        p_dq += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        if USE_G:
            p_g += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H)
        if USE_GK:
            p_gk += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        if USE_GV:
            p_gv += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * V

    # sync threads
    tl.debug_barrier()

    if HEAD_FIRST:
        p_q = q + i_nh * T*K + ((T - 1) * K if not REVERSE else 0) + i_k * BK + tl.arange(0, BK)
        p_k = k + i_nh * T*K + ((T - 1) * K if not REVERSE else 0) + i_k * BK + tl.arange(0, BK)
        p_v = v + i_nh * T*V + ((T - 1) * V if not REVERSE else 0) + i_v * BV + tl.arange(0, BV)
        p_do = do + i_nh * T*V + ((T - 1) * V if not REVERSE else 0) + i_v * BV + tl.arange(0, BV)
        p_dk = dk + (i_v * B*H + i_nh) * T*K + ((T - 1) * K if not REVERSE else 0) + i_k * BK + tl.arange(0, BK)
        p_dv = dv + (i_k * B*H + i_nh) * T*V + ((T - 1) * V if not REVERSE else 0) + i_v * BV + tl.arange(0, BV)
        if USE_G:
            p_g = g + i_nh * T + ((T - 1) if not REVERSE else 0)
        if USE_GK:
            p_gk = gk + i_nh * T*K + ((T - 1) * K if not REVERSE else 0) + i_k * BK + tl.arange(0, BK)
        if USE_GV:
            p_gv = gv + i_nh * T*V + ((T - 1) * V if not REVERSE else 0) + i_v * BV + tl.arange(0, BV)
    else:
        p_q = q + (bos + ((T - 1) if not REVERSE else 0)) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_k = k + (bos + ((T - 1) if not REVERSE else 0)) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_v = v + (bos + ((T - 1) if not REVERSE else 0)) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)
        p_do = do + (bos + ((T - 1) if not REVERSE else 0)) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)
        p_dk = dk + ((i_v * all + bos) + ((T - 1) if not REVERSE else 0)) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_dv = dv + ((i_k * all + bos) + ((T - 1) if not REVERSE else 0)) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)
        if USE_G:
            p_g = g + (bos + ((T - 1) if not REVERSE else 0)) * H + i_h
        if USE_GK:
            p_gk = gk + (bos + ((T - 1) if not REVERSE else 0)) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        if USE_GV:
            p_gv = gv + (bos + ((T - 1) if not REVERSE else 0)) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = dht + i_nh * K*V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_dh += tl.load(p_dht, mask=mask_h, other=0).to(tl.float32)

    for _ in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        b_dh += b_q[:, None] * b_do[None, :]
        b_dk = tl.sum(b_dh * b_v[None, :], axis=1)
        b_dv = tl.sum(b_dh * b_k[:, None], axis=0)
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_dh *= tl.exp(b_g)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
            b_dh *= tl.exp(b_gk)[:, None]
        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_v, other=0).to(tl.float32)
            b_dh *= tl.exp(b_gv)[None, :]
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=mask_v)

        p_q += (1 if REVERSE else -1) * (1 if HEAD_FIRST else H) * K
        p_k += (1 if REVERSE else -1) * (1 if HEAD_FIRST else H) * K
        p_v += (1 if REVERSE else -1) * (1 if HEAD_FIRST else H) * V
        p_do += (1 if REVERSE else -1) * (1 if HEAD_FIRST else H) * V
        p_dk += (1 if REVERSE else -1) * (1 if HEAD_FIRST else H) * K
        p_dv += (1 if REVERSE else -1) * (1 if HEAD_FIRST else H) * V
        if USE_G:
            p_g += (1 if REVERSE else -1) * (1 if HEAD_FIRST else H)
        if USE_GK:
            p_gk += (1 if REVERSE else -1) * (1 if HEAD_FIRST else H) * K
        if USE_GV:
            p_gv += (1 if REVERSE else -1) * (1 if HEAD_FIRST else H) * V

    if STORE_INITIAL_STATE_GRADIENT:
        p_dh0 = dh0 + i_nh * K*V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=mask_h)


def fused_recurrent_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    gv: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    reverse: bool = False,
    offsets: Optional[torch.LongTensor] = None,
    head_first: bool = True
):
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if offsets is None else len(offsets) - 1
    BK, BV = min(K, 64), min(V, 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    h0 = initial_state
    if output_final_state:
        ht = q.new_empty(N, H, K, V, dtype=torch.float32)
    else:
        ht = None
    o = q.new_empty(NK, *v.shape, dtype=torch.float32)

    grid = (NV, NK, N * H)
    fused_recurrent_fwd_kernel[grid](
        q,
        k,
        v,
        g,
        gk,
        gv,
        o,
        h0,
        ht,
        offsets,
        scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        REVERSE=reverse,
        HEAD_FIRST=head_first
    )
    o = o.sum(0)
    return o, ht


def fused_recurrent_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    gv: Optional[torch.Tensor] = None,
    o: Optional[torch.Tensor] = None,
    do: Optional[torch.Tensor] = None,
    dht: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    reverse: bool = False,
    offsets: Optional[torch.LongTensor] = None,
    head_first: bool = True
):
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if offsets is None else len(offsets) - 1

    BK, BV = min(K, 64), min(V, 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    dq = q.new_empty(NV, *q.shape, dtype=torch.float32)
    dk = q.new_empty(NV, *k.shape, dtype=torch.float32)
    dv = q.new_empty(NK, *v.shape, dtype=torch.float32)
    h0 = initial_state
    dh0 = torch.empty_like(initial_state) if initial_state is not None else None

    grid = (NV, NK, N * H)
    fused_recurrent_bwd_kernel[grid](
        q,
        k,
        v,
        g,
        gk,
        gv,
        h0,
        do,
        dq,
        dk,
        dv,
        dht,
        dh0,
        offsets,
        scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        REVERSE=reverse,
        HEAD_FIRST=head_first
    )
    dq = dq.sum(0)
    dk = dk.sum(0)
    dv = dv.sum(0)
    dg, dgk, dgv = None, None, None
    if g is not None:
        dg = chunk_global_cumsum(
            (dq * q.float() - dk * k.float()).sum(-1),
            reverse=not reverse,
            offsets=offsets,
            head_first=head_first
        )
    if gk is not None:
        dgk = chunk_global_cumsum(
            dq * q.float() - dk * k.float(),
            reverse=not reverse,
            offsets=offsets,
            head_first=head_first
        )
    if gv is not None:
        dgv = chunk_global_cumsum(
            do.float() * o.float() - dv * v.float(),
            reverse=not reverse,
            offsets=offsets,
            head_first=head_first
        )

    return dq, dk, dv, dg, dgk, dgv, dh0


class FusedRecurrentFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        gk: Optional[torch.Tensor] = None,
        gv: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        reverse: bool = False,
        offsets: Optional[torch.LongTensor] = None,
        head_first: bool = True
    ):
        o, ht = fused_recurrent_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            gk=gk,
            gv=gv,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            reverse=reverse,
            offsets=offsets,
            head_first=head_first
        )
        ctx.save_for_backward(q, k, v, g, gk, gv, initial_state, o)
        ctx.scale = scale
        ctx.reverse = reverse
        ctx.offsets = offsets
        ctx.head_first = head_first
        return o.to(q.dtype), ht

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        q, k, v, g, gk, gv, initial_state, o = ctx.saved_tensors
        # not supported yet.
        if dht is not None:
            if not dht.eq(0).all():
                if g is not None:
                    assert g.requires_grad is False, "Cannot load final state gradient and use gates at the same time"
                if gk is not None:
                    assert gk.requires_grad is False, "Cannot load final state gradient and use gates at the same time"
                if gv is not None:
                    assert gv.requires_grad is False, "Cannot load final state gradient and use gates at the same time"
        dq, dk, dv, dg, dgk, dgv, dh0 = fused_recurrent_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            gk=gk,
            gv=gv,
            o=o,
            do=do,
            dht=dht,
            scale=ctx.scale,
            initial_state=initial_state,
            reverse=ctx.reverse,
            offsets=ctx.offsets,
            head_first=ctx.head_first
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dg, dgk, dgv, None, dh0, None, None, None, None


def fused_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    gv: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = True
):
    if scale is None:
        scale = 1 # k.shape[-1] ** -0.5
    return FusedRecurrentFunction.apply(
        q,
        k,
        v,
        g,
        gk,
        gv,
        scale,
        initial_state,
        output_final_state,
        reverse,
        cu_seqlens,
        head_first
    )

def fused_recurrent_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: Optional[torch.Tensor] = None,
    gv: Optional[torch.Tensor] = None,
    scale: Optional[int] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        gk (torch.Tensor):
            Forget gates of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]` applied to keys.
        gv (torch.Tensor):
            Forget gates of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]` applied to values.
        scale (Optional[int]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        reverse (Optional[bool]):
            If `True`, process the state passing in reverse order. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gla import fused_recurrent_gla
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = torch.randn(B, T, H, K, device='cuda')
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> g = F.logsigmoid(torch.randn(B, T, H, K, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, device='cuda')
        >>> o, ht = fused_recurrent_gla(q, k, v, g,
                                        initial_state=h0,
                                        output_final_state=True,
                                        head_first=False)
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g = map(lambda x: rearrange(x, 'b t h d -> 1 (b t) h d'), (q, k, v, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = fused_recurrent_gla(q, k, v, g,
                                                initial_state=h0,
                                                output_final_state=True,
                                                cu_seqlens=cu_seqlens,
                                                head_first=False)
        >>> assert o.allclose(o_var.view(o.shape))
        >>> assert ht.allclose(ht_var)
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                             f"Please flatten variable-length inputs before processing.")
        if head_first:
            raise RuntimeError("Sequences with variable lengths are not supported for head-first mode")
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(f"The number of initial states is expected to be equal to the number of input sequences, "
                             f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.")
    if scale is None:
        scale = 1 # k.shape[-1] ** -0.5
    o, final_state = fused_recurrent(
        q=q,
        k=k,
        v=v,
        g=None,
        gk=gk,
        gv=gv,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        head_first=head_first
    )
    return o, final_state