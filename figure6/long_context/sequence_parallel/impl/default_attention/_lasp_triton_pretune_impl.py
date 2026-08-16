import torch
import torch.distributed as dist
import triton
import triton.language as tl
from einops import rearrange

from meepo.cuda import jit_cache

BLOCK = 256
CBLOCK = 64


@jit_cache(("n",))
@triton.jit
def _fwd_diag_kernel(
    Q,
    K,
    V,
    Out,
    S,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    # NUM_CBLOCK: tl.constexpr,
):
    NUM_CBLOCK: tl.constexpr = BLOCK // CBLOCK
    off = tl.program_id(0)
    off_bh = off // NUM_BLOCK
    off_block = off % NUM_BLOCK
    off_cblock = tl.program_id(1)

    off_h = off_bh % h

    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e

    block_offset = off_block * BLOCK
    qk_block_offset = block_offset * d
    v_block_offset = block_offset * e
    o_block_offset = block_offset * e

    cblock_offset = off_cblock * CBLOCK
    q_cblock_offset = cblock_offset * d
    o_cblock_offset = cblock_offset * e

    Q_block_ptr = (
        Q
        + qk_offset
        + qk_block_offset
        + q_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    K_trans_block_ptr = (
        K
        + qk_offset
        + qk_block_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )
    V_block_ptr = (
        V
        + v_offset
        + v_block_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )
    O_block_ptr = (
        Out
        + o_offset
        + o_block_offset
        + o_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )

    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    i = off_cblock
    q_index = tl.arange(0, CBLOCK) + i * CBLOCK

    q = tl.load(Q_block_ptr, mask=q_index[:, None] < n, other=0.0).to(
        tl.float32
    )

    qkv = tl.zeros([CBLOCK, e], dtype=tl.float32)

    for j in range(i + 1):
        kv_index = tl.arange(0, CBLOCK) + j * CBLOCK
        diff = q_index[:, None] - kv_index[None, :]
        s_index = s * diff
        s_index = tl.where(diff >= 0, -s_index, float("-inf"))
        decay = tl.exp(s_index)

        k_trans = tl.load(
            K_trans_block_ptr, mask=kv_index[None, :] < n, other=0.0
        ).to(tl.float32)
        v = tl.load(V_block_ptr, mask=kv_index[:, None] < n, other=0.0).to(
            tl.float32
        )

        qk = tl.dot(q, k_trans) * decay

        qkv += tl.dot(qk, v)

        K_trans_block_ptr += CBLOCK * d
        V_block_ptr += CBLOCK * e

    tl.store(
        O_block_ptr,
        qkv.to(O_block_ptr.dtype.element_ty),
        mask=q_index[:, None] < n,
    )


@jit_cache(("n",))
@triton.jit
def _fwd_kv_parallel(
    K,
    V,
    S,
    KV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    # D_FBLOCK: tl.constexpr,
    # E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    # NUM_CBLOCK: tl.constexpr,
):
    D_FBLOCK: tl.constexpr = d // NUM_FBLOCK
    E_FBLOCK: tl.constexpr = e // NUM_FBLOCK
    NUM_CBLOCK: tl.constexpr = BLOCK // CBLOCK
    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    off_de = tl.program_id(2)

    off_h = off_bh % h
    off_d = off_de // NUM_FBLOCK
    off_e = off_de % NUM_FBLOCK

    block_offset = off_block * BLOCK

    k_block_offset = block_offset * d
    v_block_offset = block_offset * e
    kv_block_offset = off_block * d * e

    k_offset = off_bh * n * d
    v_offset = off_bh * n * e
    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK

    # (CBLOCK, FBLOCK)
    K_trans_block_ptr = (
        K
        + k_offset
        + k_block_offset
        + d_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, D_FBLOCK)[:, None]
    )
    V_block_ptr = (
        V
        + v_offset
        + v_block_offset
        + e_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )
    KV_block_ptr = (
        KV
        + kv_offset
        + kv_block_offset
        + d_offset * e
        + e_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    s_ptrs = S + off_h
    s = tl.load(s_ptrs)

    # compute block array
    c_array = tl.arange(0, CBLOCK)

    kv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)
    for j in range(NUM_CBLOCK):
        k_trans = tl.load(K_trans_block_ptr).to(tl.float32)
        v = tl.load(V_block_ptr).to(tl.float32)
        k_decay = tl.exp(
            -s.to(tl.float32) * (BLOCK - (j * CBLOCK + c_array[None, :]))
        )

        kv += tl.dot(k_trans * k_decay, v)

        K_trans_block_ptr += CBLOCK * d
        V_block_ptr += CBLOCK * e

    tl.store(KV_block_ptr, kv.to(KV_block_ptr.dtype.element_ty))


@jit_cache(("n",))
@triton.jit
def _fwd_kv_sum(
    K,
    V,
    S,
    KV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    # D_FBLOCK: tl.constexpr,
    # E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    # NUM_CBLOCK: tl.constexpr,
):
    D_FBLOCK: tl.constexpr = d // NUM_FBLOCK
    E_FBLOCK: tl.constexpr = e // NUM_FBLOCK
    NUM_CBLOCK: tl.constexpr = BLOCK // CBLOCK
    off_bh = tl.program_id(0)
    off_h = off_bh % h
    off_d = tl.program_id(1)
    off_e = tl.program_id(2)

    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK

    # (CBLOCK, FBLOCK)
    KV_block_ptr = (
        KV
        + kv_offset
        + d_offset * e
        + e_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    s_ptrs = S + off_h
    s = tl.load(s_ptrs)

    block_decay = tl.exp(-s.to(tl.float32) * BLOCK)

    # compute block array

    kv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)
    for i in range(NUM_BLOCK):
        kv_current = tl.load(KV_block_ptr).to(tl.float32)

        kv = block_decay * kv + kv_current
        KV_block_ptr += d * e

    # for GKV
    tl.store(KV_block_ptr, kv.to(KV_block_ptr.dtype.element_ty))


@jit_cache(("n",))
@triton.jit
def _fwd_kv_reduce(
    K,
    V,
    S,
    KV,
    GKV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    # D_FBLOCK: tl.constexpr,
    # E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    # CBLOCK: tl.constexpr,
    # NUM_CBLOCK: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_h = off_bh % h
    off_d = tl.program_id(1)
    off_e = tl.program_id(2)
    D_FBLOCK: tl.constexpr = d // NUM_FBLOCK
    E_FBLOCK: tl.constexpr = e // NUM_FBLOCK

    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e
    gkv_offset = off_bh * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK

    # (CBLOCK, FBLOCK)
    KV_block_ptr = (
        KV
        + kv_offset
        + d_offset * e
        + e_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )
    GKV_block_ptr = (
        GKV
        + gkv_offset
        + d_offset * e
        + e_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    s_ptrs = S + off_h
    s = tl.load(s_ptrs)

    block_decay = tl.exp(-s.to(tl.float32) * BLOCK)

    # compute block array

    kv = tl.load(GKV_block_ptr).to(tl.float32)
    for i in range(NUM_BLOCK):
        kv_current = tl.load(KV_block_ptr).to(tl.float32)
        tl.store(KV_block_ptr, kv.to(KV_block_ptr.dtype.element_ty))

        kv = block_decay * kv + kv_current
        KV_block_ptr += d * e


@jit_cache(("n",))
@triton.jit
def _fwd_none_diag_kernel(
    Q,
    K,
    V,
    Out,
    S,
    KV,
    GKV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    # D_FBLOCK: tl.constexpr,
    # E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    # NUM_CBLOCK: tl.constexpr,
):
    D_FBLOCK: tl.constexpr = d // NUM_FBLOCK
    E_FBLOCK: tl.constexpr = e // NUM_FBLOCK
    NUM_CBLOCK: tl.constexpr = BLOCK // CBLOCK
    off_bh = tl.program_id(0)
    off_h = off_bh % h

    off_nc = tl.program_id(1)
    off_n = off_nc // NUM_CBLOCK
    off_c = off_nc % NUM_CBLOCK
    off_e = tl.program_id(2)

    n_offset = off_n * BLOCK
    c_offset = off_c * CBLOCK
    e_offset = off_e * E_FBLOCK

    q_offset = off_bh * n * d + (n_offset + c_offset) * d
    o_offset = off_bh * n * e + (n_offset + c_offset) * e + e_offset

    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e + off_n * d * e + e_offset
    # gkv_offset = off_bh * d * e + e_offset

    Q_block_ptr = (
        Q
        + q_offset
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    O_block_ptr = (
        Out
        + o_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )
    KV_block_ptr = (
        KV
        + kv_offset
        + tl.arange(0, d)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )
    # GKV_block_ptr = (
    #     GKV
    #     + gkv_offset
    #     + tl.arange(0, d)[:, None] * e
    #     + tl.arange(0, E_FBLOCK)[None, :]
    # )

    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    c_array = tl.arange(0, CBLOCK)

    # GKV = tl.load(GKV_block_ptr).to(tl.float32)
    kv = tl.load(KV_block_ptr).to(tl.float32)
    q = tl.load(Q_block_ptr).to(tl.float32)
    q_decay = tl.exp(-s.to(tl.float32) * (c_offset + c_array[:, None]))
    # qkv_none_diag = tl.dot(q, kv) * q_decay + tl.dot(q, GKV) * tl.exp(
    #     -s.to(tl.float32) * (c_offset + c_array[:, None] + n_offset)
    # )
    # qkv_none_diag = tl.dot(q, kv + GKV * tl.exp(-s.to(tl.float32) * n_offset)) * q_decay
    qkv_none_diag = tl.dot(q, kv) * q_decay
    qkv_diag = tl.load(O_block_ptr).to(tl.float32)

    qkv = qkv_diag + qkv_none_diag

    tl.store(O_block_ptr, qkv.to(O_block_ptr.dtype.element_ty))


@jit_cache(("n",))
@triton.jit
def _bwd_diag_kernel(
    Q,
    K,
    V,
    S,
    DO,
    DQ,
    DK,
    DV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    # NUM_CBLOCK: tl.constexpr,
):
    # D_FBLOCK: tl.constexpr = d // NUM_FBLOCK
    # E_FBLOCK: tl.constexpr = e // NUM_FBLOCK
    NUM_CBLOCK: tl.constexpr = BLOCK // CBLOCK
    off = tl.program_id(0)
    off_bh = off // NUM_BLOCK
    off_block = off % NUM_BLOCK
    off_cblock = tl.program_id(1)

    off_h = off_bh % h

    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e

    block_offset = off_block * BLOCK
    qk_block_offset = block_offset * d
    v_block_offset = block_offset * e
    o_block_offset = block_offset * e

    cblock_offset = off_cblock * CBLOCK
    qk_cblock_offset = cblock_offset * d
    v_cblock_offset = cblock_offset * e
    o_cblock_offset = cblock_offset * e

    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    # dq
    DO_block_ptr = (
        DO
        + o_offset
        + o_block_offset
        + o_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )
    DQ_block_ptr = (
        DQ
        + qk_offset
        + qk_block_offset
        + qk_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    K_block_ptr = (
        K
        + qk_offset
        + qk_block_offset
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    V_trans_block_ptr = (
        V
        + v_offset
        + v_block_offset
        + tl.arange(0, CBLOCK)[None, :] * e
        + tl.arange(0, e)[:, None]
    )

    do = tl.load(DO_block_ptr).to(tl.float32)
    dq = tl.zeros([CBLOCK, d], dtype=tl.float32)

    i = off_cblock
    do_index = tl.arange(0, CBLOCK) + i * CBLOCK
    for j in range(i + 1):
        k = tl.load(K_block_ptr).to(tl.float32)
        v_trans = tl.load(V_trans_block_ptr).to(tl.float32)

        # compute
        v_index = tl.arange(0, CBLOCK) + j * CBLOCK
        diff = do_index[:, None] - v_index[None, :]
        s_index = s * diff
        s_index = tl.where(diff >= 0, -s_index, float("-inf"))
        diag_decay = tl.exp(s_index)

        dqk = tl.dot(do, v_trans) * diag_decay
        dq += tl.dot(dqk, k)

        K_block_ptr += CBLOCK * d
        V_trans_block_ptr += CBLOCK * e

    tl.store(DQ_block_ptr, dq.to(DQ_block_ptr.dtype.element_ty))

    # dk
    V_trans_block_ptr = (
        V
        + v_offset
        + v_block_offset
        + v_cblock_offset
        + tl.arange(0, CBLOCK)[None, :] * e
        + tl.arange(0, e)[:, None]
    )
    DO_block_ptr = (
        DO
        + o_offset
        + o_block_offset
        + o_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )
    Q_trans_block_ptr = (
        Q
        + qk_offset
        + qk_block_offset
        + qk_cblock_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )
    DK_trans_block_ptr = (
        DK
        + qk_offset
        + qk_block_offset
        + qk_cblock_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )

    v_trans = tl.load(V_trans_block_ptr).to(tl.float32)
    v_index = tl.arange(0, CBLOCK) + i * CBLOCK
    dk_trans = tl.zeros([d, CBLOCK], dtype=tl.float32)

    # add
    K_block_ptr = (
        K
        + qk_offset
        + qk_block_offset
        + qk_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    DV_block_ptr = (
        DV
        + v_offset
        + v_block_offset
        + v_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )

    dv = tl.zeros([CBLOCK, e], dtype=tl.float32)
    k = tl.load(K_block_ptr).to(tl.float32)
    for j in range(i, NUM_CBLOCK):
        q_trans = tl.load(Q_trans_block_ptr).to(tl.float32)
        do = tl.load(DO_block_ptr).to(tl.float32)

        do_index = tl.arange(0, CBLOCK) + j * CBLOCK
        diff = do_index[:, None] - v_index[None, :]
        s_index = s * diff
        s_index = tl.where(diff >= 0, -s_index, float("-inf"))
        diag_decay = tl.exp(s_index)

        dqk = tl.dot(do, v_trans) * diag_decay
        dk_trans += tl.dot(q_trans, dqk)

        Q_trans_block_ptr += CBLOCK * d
        DO_block_ptr += CBLOCK * e

        # add
        diag_decay_trans = tl.trans(diag_decay)
        qk_trans = tl.dot(k, q_trans) * diag_decay_trans
        dv += tl.dot(qk_trans, do)

    tl.store(
        DK_trans_block_ptr, dk_trans.to(DK_trans_block_ptr.dtype.element_ty)
    )
    tl.store(DV_block_ptr, dv.to(DV_block_ptr.dtype.element_ty))


@jit_cache(("n",))
@triton.jit
def _bwd_dkv_parallel(
    Q,
    DO,
    S,
    DKV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    # D_FBLOCK: tl.constexpr,
    # E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    # NUM_CBLOCK: tl.constexpr,
):
    D_FBLOCK: tl.constexpr = d // NUM_FBLOCK
    E_FBLOCK: tl.constexpr = e // NUM_FBLOCK
    NUM_CBLOCK: tl.constexpr = BLOCK // CBLOCK
    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    off_de = tl.program_id(2)

    off_h = off_bh % h
    off_d = off_de // NUM_FBLOCK
    off_e = off_de % NUM_FBLOCK

    block_offset = off_block * BLOCK
    qk_block_offset = block_offset * d
    o_block_offset = block_offset * e
    kv_block_offset = off_block * d * e

    qk_offset = off_bh * n * d
    o_offset = off_bh * n * e
    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK

    # (CBLOCK, FBLOCK)
    DKV_block_ptr = (
        DKV
        + kv_offset
        + kv_block_offset
        + d_offset * e
        + e_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    Q_trans_block_ptr = (
        Q
        + qk_offset
        + qk_block_offset
        + d_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, D_FBLOCK)[:, None]
    )
    DO_block_ptr = (
        DO
        + o_offset
        + o_block_offset
        + e_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    s_ptrs = S + off_h
    s = tl.load(s_ptrs)

    c_array = tl.arange(0, CBLOCK)

    dkv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)

    for j in range(NUM_CBLOCK):
        do = tl.load(DO_block_ptr).to(tl.float32)
        q_trans = tl.load(Q_trans_block_ptr).to(tl.float32)
        q_decay_trans = tl.exp(
            -s.to(tl.float32) * (j * CBLOCK + c_array[None, :])
        )
        dkv += tl.dot(q_trans * q_decay_trans, do)

        DO_block_ptr += CBLOCK * e
        Q_trans_block_ptr += CBLOCK * d

    tl.store(DKV_block_ptr, dkv.to(DKV_block_ptr.dtype.element_ty))


@jit_cache(("n",))
@triton.jit
def _bwd_dkv_sum(
    Q,
    DO,
    S,
    DKV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    # D_FBLOCK: tl.constexpr,
    # E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    # CBLOCK: tl.constexpr,
    # NUM_CBLOCK: tl.constexpr,
):
    D_FBLOCK: tl.constexpr = d // NUM_FBLOCK
    E_FBLOCK: tl.constexpr = e // NUM_FBLOCK
    # NUM_CBLOCK: tl.constexpr = BLOCK // CBLOCK
    off_bh = tl.program_id(0)
    off_h = off_bh % h
    off_d = tl.program_id(1)
    off_e = tl.program_id(2)

    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK

    DKV_block_ptr = (
        DKV
        + kv_offset
        + d_offset * e
        + e_offset
        + NUM_BLOCK * d * e
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    s_ptrs = S + off_h
    s = tl.load(s_ptrs)

    block_decay = tl.exp(-s.to(tl.float32) * BLOCK)

    # compute block array

    dkv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)
    for i in range(NUM_BLOCK - 1, -1, -1):
        DKV_block_ptr -= d * e
        dkv_current = tl.load(DKV_block_ptr).to(tl.float32)

        dkv = block_decay * dkv + dkv_current

    # store at last pos
    DKV_block_ptr += NUM_BLOCK * d * e
    tl.store(DKV_block_ptr, dkv.to(DKV_block_ptr.dtype.element_ty))


@jit_cache(("n",))
@triton.jit
def _bwd_dkv_reduce(
    Q,
    DO,
    S,
    DKV,
    GDKV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    # D_FBLOCK: tl.constexpr,
    # E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    # CBLOCK: tl.constexpr,
    # NUM_CBLOCK: tl.constexpr,
):
    D_FBLOCK: tl.constexpr = d // NUM_FBLOCK
    E_FBLOCK: tl.constexpr = e // NUM_FBLOCK
    # NUM_CBLOCK: tl.constexpr = BLOCK // CBLOCK
    off_bh = tl.program_id(0)
    off_h = off_bh % h
    off_d = tl.program_id(1)
    off_e = tl.program_id(2)

    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e
    gkv_offset = off_bh * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK

    DKV_block_ptr = (
        DKV
        + kv_offset
        + d_offset * e
        + e_offset
        + NUM_BLOCK * d * e
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )
    GDKV_block_ptr = (
        GDKV
        + gkv_offset
        + d_offset * e
        + e_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    s_ptrs = S + off_h
    s = tl.load(s_ptrs)

    block_decay = tl.exp(-s.to(tl.float32) * BLOCK)

    # compute block array

    dkv = tl.load(GDKV_block_ptr).to(tl.float32)
    for i in range(NUM_BLOCK - 1, -1, -1):
        DKV_block_ptr -= d * e
        dkv_current = tl.load(DKV_block_ptr).to(tl.float32)
        tl.store(DKV_block_ptr, dkv.to(DKV_block_ptr.dtype.element_ty))

        dkv = block_decay * dkv + dkv_current


@jit_cache(("n",))
@triton.jit
def _bwd_none_diag_kernel(
    Q,
    K,
    V,
    S,
    DO,
    DQ,
    DK,
    DV,
    KV,
    DKV,
    GKV,
    GDKV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    # D_FBLOCK: tl.constexpr,
    # E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    # NUM_CBLOCK: tl.constexpr,
):
    D_FBLOCK: tl.constexpr = d // NUM_FBLOCK
    E_FBLOCK: tl.constexpr = e // NUM_FBLOCK
    NUM_CBLOCK: tl.constexpr = BLOCK // CBLOCK
    off_bh = tl.program_id(0)
    off_h = off_bh % h

    off_nc = tl.program_id(1)
    off_n = off_nc // NUM_CBLOCK
    off_c = off_nc % NUM_CBLOCK
    off_de = tl.program_id(2)

    n_offset = off_n * BLOCK
    c_offset = off_c * CBLOCK
    d_offset = off_de * D_FBLOCK
    e_offset = off_de * E_FBLOCK

    qk_offset = off_bh * n * d + (n_offset + c_offset) * d
    v_offset = off_bh * n * e + (n_offset + c_offset) * e
    o_offset = off_bh * n * e + (n_offset + c_offset) * e

    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e + off_n * d * e
    kv_trans_offset = off_bh * (NUM_BLOCK + 1) * d * e + off_n * d * e

    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    # dq
    DO_block_ptr = (
        DO
        + o_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )
    DQ_block_ptr = (
        DQ
        + qk_offset
        + d_offset
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, D_FBLOCK)[None, :]
    )
    KV_trans_block_ptr = (
        KV
        + kv_trans_offset
        + d_offset * e
        + tl.arange(0, D_FBLOCK)[None, :] * e
        + tl.arange(0, e)[:, None]
    )

    c_array = tl.arange(0, CBLOCK)
    kv_trans = tl.load(KV_trans_block_ptr).to(tl.float32)
    q_decay = tl.exp(-s.to(tl.float32) * (c_offset + c_array[:, None]))
    do = tl.load(DO_block_ptr).to(tl.float32)
    dq_none_diag = tl.dot(do, kv_trans) * q_decay
    dq = dq_none_diag + tl.load(DQ_block_ptr)
    tl.store(DQ_block_ptr, dq.to(DQ_block_ptr.dtype.element_ty))

    # dk
    DK_trans_block_ptr = (
        DK
        + qk_offset
        + d_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, D_FBLOCK)[:, None]
    )
    V_trans_block_ptr = (
        V
        + v_offset
        + tl.arange(0, CBLOCK)[None, :] * e
        + tl.arange(0, e)[:, None]
    )
    DKV_block_ptr = (
        DKV
        + kv_offset
        + d_offset * e
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )

    v_trans = tl.load(V_trans_block_ptr).to(tl.float32)
    dkv = tl.load(DKV_block_ptr).to(tl.float32)
    k_decay_trans = tl.exp(
        -s.to(tl.float32) * (BLOCK - (off_c * CBLOCK + c_array[None, :]))
    )

    # !!! important !!!
    dk_none_diag_trans = tl.dot(dkv, v_trans) * k_decay_trans

    dk_trans = dk_none_diag_trans + tl.load(DK_trans_block_ptr)
    tl.store(
        DK_trans_block_ptr, dk_trans.to(DK_trans_block_ptr.dtype.element_ty)
    )

    # dv
    DKV_block_ptr_ = (
        DKV
        + kv_offset
        + e_offset
        + tl.arange(0, d)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )
    K_block_ptr = (
        K
        + qk_offset
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    DV_block_ptr = (
        DV
        + v_offset
        + e_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    k_decay = tl.exp(
        -s.to(tl.float32) * (BLOCK - (off_c * CBLOCK + c_array[:, None]))
    )
    k = tl.load(K_block_ptr).to(tl.float32)

    dkv_ = tl.load(DKV_block_ptr_).to(tl.float32)
    dv_none_diag = tl.dot(k, dkv_) * k_decay
    dv = dv_none_diag + tl.load(DV_block_ptr)
    tl.store(DV_block_ptr, dv.to(DV_block_ptr.dtype.element_ty))


@jit_cache(("n",))
@triton.jit
def _fwd_kv_parallel_sum(
    K,
    V,
    S,
    KV,
    kv_parallel_cnt,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    # D_FBLOCK: tl.constexpr,
    # E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    # NUM_CBLOCK: tl.constexpr,
):
    D_FBLOCK: tl.constexpr = d // NUM_FBLOCK
    E_FBLOCK: tl.constexpr = e // NUM_FBLOCK
    NUM_CBLOCK: tl.constexpr = BLOCK // CBLOCK
    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    off_de = tl.program_id(2)

    off_h = off_bh % h
    off_d = off_de // NUM_FBLOCK
    off_e = off_de % NUM_FBLOCK

    block_offset = off_block * BLOCK

    k_block_offset = block_offset * d
    v_block_offset = block_offset * e
    kv_block_offset = off_block * d * e

    k_offset = off_bh * n * d
    v_offset = off_bh * n * e
    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK

    # (CBLOCK, FBLOCK)
    K_trans_block_ptr = (
        K
        + k_offset
        + k_block_offset
        + d_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, D_FBLOCK)[:, None]
    )
    V_block_ptr = (
        V
        + v_offset
        + v_block_offset
        + e_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )
    KV_block_ptr = (
        KV
        + kv_offset
        + kv_block_offset
        + d_offset * e
        + e_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )
    s_ptrs = S + off_h
    s = tl.load(s_ptrs)

    # compute block array
    c_array = tl.arange(0, CBLOCK)

    kv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)
    for j in range(NUM_CBLOCK):
        k_trans = tl.load(K_trans_block_ptr).to(tl.float32)
        v = tl.load(V_block_ptr).to(tl.float32)
        k_decay = tl.exp(
            -s.to(tl.float32) * (BLOCK - (j * CBLOCK + c_array[None, :]))
        )

        kv += tl.dot(k_trans * k_decay, v)

        K_trans_block_ptr += CBLOCK * d
        V_block_ptr += CBLOCK * e

    tl.store(KV_block_ptr, kv.to(KV_block_ptr.dtype.element_ty))

    tl.debug_barrier()
    # Atomically increment the number of finished blocks
    block_finished_flag = kv_parallel_cnt + off_bh
    finished_blocks = tl.atomic_add(block_finished_flag, 1)
    # Check if this is the last block
    is_last_block = finished_blocks == (NUM_BLOCK - 1)
    if not is_last_block:
        return
    # ... execute code for the last block ...
    block_decay = tl.exp(-s.to(tl.float32) * BLOCK)

    # compute block array

    reduce_KV_block_ptr = (
        KV
        + kv_offset
        + d_offset * e
        + e_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )
    reduce_kv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)
    for i in range(NUM_BLOCK):
        kv_current = tl.load(reduce_KV_block_ptr).to(tl.float32)

        reduce_kv = block_decay * reduce_kv + kv_current
        reduce_KV_block_ptr += d * e

    # for GKV
    tl.store(
        reduce_KV_block_ptr, reduce_kv.to(reduce_KV_block_ptr.dtype.element_ty)
    )


@triton.jit
def _bwd_dkv_parallel_sum(
    Q,
    DO,
    S,
    DKV,
    dkv_parallel_cnt,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    off_de = tl.program_id(2)

    off_h = off_bh % h
    off_d = off_de // NUM_FBLOCK
    off_e = off_de % NUM_FBLOCK

    block_offset = off_block * BLOCK
    qk_block_offset = block_offset * d
    o_block_offset = block_offset * e
    kv_block_offset = off_block * d * e

    qk_offset = off_bh * n * d
    o_offset = off_bh * n * e
    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK

    # (CBLOCK, FBLOCK)
    DKV_block_ptr = (
        DKV
        + kv_offset
        + kv_block_offset
        + d_offset * e
        + e_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    Q_trans_block_ptr = (
        Q
        + qk_offset
        + qk_block_offset
        + d_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, D_FBLOCK)[:, None]
    )
    DO_block_ptr = (
        DO
        + o_offset
        + o_block_offset
        + e_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    s_ptrs = S + off_h
    s = tl.load(s_ptrs)

    c_array = tl.arange(0, CBLOCK)

    dkv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)

    for j in range(NUM_CBLOCK):
        do = tl.load(DO_block_ptr).to(tl.float32)
        q_trans = tl.load(Q_trans_block_ptr).to(tl.float32)
        q_decay_trans = tl.exp(
            -s.to(tl.float32) * (j * CBLOCK + c_array[None, :])
        )
        dkv += tl.dot(q_trans * q_decay_trans, do)

        DO_block_ptr += CBLOCK * e
        Q_trans_block_ptr += CBLOCK * d

    tl.store(DKV_block_ptr, dkv.to(DKV_block_ptr.dtype.element_ty))

    tl.debug_barrier()
    # Atomically increment the number of finished blocks
    block_finished_flag = dkv_parallel_cnt + off_bh
    finished_blocks = tl.atomic_add(block_finished_flag, 1)
    # Check if this is the last block
    is_last_block = finished_blocks == (NUM_BLOCK - 1)
    if not is_last_block:
        return
    # ... execute code for the last block ...
    block_decay = tl.exp(-s.to(tl.float32) * BLOCK)

    # compute block array
    reduce_DKV_block_ptr = (
        DKV
        + kv_offset
        + d_offset * e
        + e_offset
        + NUM_BLOCK * d * e
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    reduce_dkv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)
    for i in range(NUM_BLOCK - 1, -1, -1):
        reduce_DKV_block_ptr -= d * e
        dkv_current = tl.load(reduce_DKV_block_ptr).to(tl.float32)

        reduce_dkv = block_decay * reduce_dkv + dkv_current

    # store at last pos
    reduce_DKV_block_ptr += NUM_BLOCK * d * e
    tl.store(
        reduce_DKV_block_ptr,
        reduce_dkv.to(reduce_DKV_block_ptr.dtype.element_ty),
    )


def proc_kv(q, k, v, s, BLOCK, CBLOCK, cp_size, use_fused_kernel):
    # shape constraints
    b, h, n, d = q.shape
    e = v.shape[-1]

    NUM_FBLOCK = 1
    D_FBLOCK = d // NUM_FBLOCK
    E_FBLOCK = e // NUM_FBLOCK
    assert d % NUM_FBLOCK == 0
    assert e % NUM_FBLOCK == 0
    grid = (b * h, NUM_FBLOCK, NUM_FBLOCK)

    NUM_BLOCK = q.shape[2] // BLOCK

    NUM_CBLOCK = BLOCK // CBLOCK
    kv = torch.empty(
        (b, h, NUM_BLOCK + 1, d, e), dtype=torch.float32, device=q.device
    )

    if use_fused_kernel:
        kv_parallel_cnt = torch.zeros((b, h), dtype=torch.int, device=q.device)

        with torch.cuda.device(q.device.index):
            # grid = (b * h, NUM_BLOCK, NUM_FBLOCK * NUM_FBLOCK)
            grid = lambda meta: (
                b * h,
                NUM_BLOCK,
                meta['NUM_FBLOCK'] * meta['NUM_FBLOCK'],
            )
            _fwd_kv_parallel_sum[grid](
                k,
                v,
                s,
                kv,
                kv_parallel_cnt,
                b,
                h,
                n,
                d,
                e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                # D_FBLOCK=D_FBLOCK,
                # E_FBLOCK=E_FBLOCK,
                # NUM_FBLOCK=NUM_FBLOCK,
                # CBLOCK=CBLOCK,
                # NUM_CBLOCK=NUM_CBLOCK,
            )
    else:
        with torch.cuda.device(q.device.index):
            # grid = (b * h, NUM_BLOCK, NUM_FBLOCK * NUM_FBLOCK)
            grid = lambda meta: (
                b * h,
                NUM_BLOCK,
                meta['NUM_FBLOCK'] * meta['NUM_FBLOCK'],
            )
            _fwd_kv_parallel[grid](
                k,
                v,
                s,
                kv,
                b,
                h,
                n,
                d,
                e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                # D_FBLOCK=D_FBLOCK,
                # E_FBLOCK=E_FBLOCK,
                # NUM_FBLOCK=NUM_FBLOCK,
                # CBLOCK=CBLOCK,
                # NUM_CBLOCK=NUM_CBLOCK,
            )

            if cp_size > 1:
                # grid = (b * h, NUM_FBLOCK, NUM_FBLOCK)
                grid = lambda meta: (
                    b * h,
                    meta['NUM_FBLOCK'],
                    meta['NUM_FBLOCK'],
                )
                _fwd_kv_sum[grid](
                    k,
                    v,
                    s,
                    kv,
                    b,
                    h,
                    n,
                    d,
                    e,
                    BLOCK=BLOCK,
                    NUM_BLOCK=NUM_BLOCK,
                    # D_FBLOCK=D_FBLOCK,
                    # E_FBLOCK=E_FBLOCK,
                    # NUM_FBLOCK=NUM_FBLOCK,
                    # CBLOCK=CBLOCK,
                    # NUM_CBLOCK=NUM_CBLOCK,
                )

    return kv


# @torch.compile(fullgraph=True, mode="reduce-overhead")
def lasp_forward(q, k, v, s, kv, KV, BLOCK=128, CBLOCK=64):
    # shape constraints
    b, h, n, d = q.shape
    e = v.shape[-1]
    # right
    o = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)

    NUM_BLOCK = q.shape[2] // BLOCK
    # NUM_CBLOCK = BLOCK // CBLOCK

    # NUM_FBLOCK = 1
    # D_FBLOCK = d // NUM_FBLOCK
    # E_FBLOCK = e // NUM_FBLOCK
    # assert d % NUM_FBLOCK == 0
    # assert e % NUM_FBLOCK == 0
    with torch.cuda.device(q.device.index):
        grid = lambda meta: (b * h, meta['NUM_FBLOCK'], meta['NUM_FBLOCK'])
        _fwd_kv_reduce[grid](
            k,
            v,
            s,
            kv,
            KV,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            # D_FBLOCK=D_FBLOCK,
            # E_FBLOCK=E_FBLOCK,
            # NUM_FBLOCK=NUM_FBLOCK,
            # CBLOCK=CBLOCK,
            # NUM_CBLOCK=NUM_CBLOCK,
        )

    with torch.cuda.device(q.device.index):
        grid = lambda meta: (b * h * NUM_BLOCK, BLOCK // meta['CBLOCK'])
        _fwd_diag_kernel[grid](
            q,
            k,
            v,
            o,
            s,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            # CBLOCK=CBLOCK,
            # NUM_CBLOCK=NUM_CBLOCK,
        )

    with torch.cuda.device(q.device.index):
        grid = lambda meta: (
            b * h,
            NUM_BLOCK * BLOCK // meta['CBLOCK'],
            meta['NUM_FBLOCK'],
        )
        _fwd_none_diag_kernel[grid](
            q,
            k,
            v,
            o,
            s,
            kv,
            KV,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            # D_FBLOCK=D_FBLOCK,
            # E_FBLOCK=E_FBLOCK,
            # NUM_FBLOCK=NUM_FBLOCK,
            # CBLOCK=CBLOCK,
            # NUM_CBLOCK=NUM_CBLOCK,
        )

    return o, KV


def proc_dkv(q, k, v, do, s, BLOCK, CBLOCK, cp_size, use_fused_kernel):
    b, h, n, d = q.shape
    e = v.shape[-1]

    # must the same as fwd
    NUM_BLOCK = n // BLOCK

    assert BLOCK % CBLOCK == 0
    NUM_CBLOCK = BLOCK // CBLOCK

    dkv = torch.empty(
        (b, h, NUM_BLOCK + 1, d, e), dtype=torch.float32, device=q.device
    )
    NUM_FBLOCK = 1
    D_FBLOCK = d // NUM_FBLOCK
    E_FBLOCK = e // NUM_FBLOCK
    assert d % NUM_FBLOCK == 0
    assert e % NUM_FBLOCK == 0

    if use_fused_kernel:
        dkv_parallel_cnt = torch.zeros((b, h), dtype=torch.int, device=q.device)
        with torch.cuda.device(q.device.index):
            grid = (b * h, NUM_BLOCK, NUM_FBLOCK * NUM_FBLOCK)
            _bwd_dkv_parallel_sum[grid](
                q,
                do,
                s,
                dkv,
                dkv_parallel_cnt,
                b,
                h,
                n,
                d,
                e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                D_FBLOCK=D_FBLOCK,
                E_FBLOCK=E_FBLOCK,
                NUM_FBLOCK=NUM_FBLOCK,
                CBLOCK=CBLOCK,
                NUM_CBLOCK=NUM_CBLOCK,
            )
    else:
        with torch.cuda.device(q.device.index):
            grid = lambda meta: (
                b * h,
                NUM_BLOCK,
                meta['NUM_FBLOCK'] * meta['NUM_FBLOCK'],
            )
            _bwd_dkv_parallel[grid](
                q,
                do,
                s,
                dkv,
                b,
                h,
                n,
                d,
                e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                # D_FBLOCK=D_FBLOCK,
                # E_FBLOCK=E_FBLOCK,
                # NUM_FBLOCK=NUM_FBLOCK,
                # CBLOCK=CBLOCK,
                # NUM_CBLOCK=NUM_CBLOCK,
            )

            if cp_size > 1:
                grid = lambda meta: (
                    b * h,
                    meta['NUM_FBLOCK'],
                    meta['NUM_FBLOCK'],
                )
                _bwd_dkv_sum[grid](
                    q,
                    do,
                    s,
                    dkv,
                    b,
                    h,
                    n,
                    d,
                    e,
                    BLOCK=BLOCK,
                    NUM_BLOCK=NUM_BLOCK,
                    # D_FBLOCK=D_FBLOCK,
                    # E_FBLOCK=E_FBLOCK,
                    # NUM_FBLOCK=NUM_FBLOCK,
                    # CBLOCK=CBLOCK,
                    # NUM_CBLOCK=NUM_CBLOCK,
                )

    return dkv


def lasp_backward(q, k, v, s, do, kv, KV, dkv, DKV, BLOCK=128, CBLOCK=64):
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    grid = (q.shape[0] * q.shape[1], 1)

    b, h, n, d = q.shape
    e = v.shape[-1]

    # must the same as fwd
    NUM_BLOCK = n // BLOCK

    assert BLOCK % CBLOCK == 0
    NUM_CBLOCK = BLOCK // CBLOCK
    NUM_FBLOCK = 1
    D_FBLOCK = d // NUM_FBLOCK
    E_FBLOCK = e // NUM_FBLOCK
    assert d % NUM_FBLOCK == 0
    assert e % NUM_FBLOCK == 0

    with torch.cuda.device(q.device.index):
        grid = lambda meta: (b * h, meta['NUM_FBLOCK'], meta['NUM_FBLOCK'])
        _bwd_dkv_reduce[grid](
            q,
            do,
            s,
            dkv,
            DKV,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            # D_FBLOCK=D_FBLOCK,
            # E_FBLOCK=E_FBLOCK,
            # NUM_FBLOCK=NUM_FBLOCK,
            # CBLOCK=CBLOCK,
            # NUM_CBLOCK=NUM_CBLOCK,
        )

    with torch.cuda.device(q.device.index):
        grid = lambda meta: (b * h * NUM_BLOCK, BLOCK // meta['CBLOCK'])
        _bwd_diag_kernel[grid](
            q,
            k,
            v,
            s,
            do,
            dq,
            dk,
            dv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            # CBLOCK=CBLOCK,
            # NUM_CBLOCK=NUM_CBLOCK,
        )

    # dkv = torch.empty((b, h, NUM_BLOCK + 1, d, e), dtype=torch.float32, device=q.device)
    NUM_FBLOCK = 1
    D_FBLOCK = d // NUM_FBLOCK
    E_FBLOCK = e // NUM_FBLOCK
    assert d % NUM_FBLOCK == 0
    assert e % NUM_FBLOCK == 0

    with torch.cuda.device(q.device.index):
        grid = lambda meta: (
            b * h,
            NUM_BLOCK * BLOCK // meta['CBLOCK'],
            meta['NUM_FBLOCK'],
        )
        _bwd_none_diag_kernel[grid](
            q,
            k,
            v,
            s,
            do,
            dq,
            dk,
            dv,
            kv,
            dkv,
            KV,
            DKV,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            # D_FBLOCK=D_FBLOCK,
            # E_FBLOCK=E_FBLOCK,
            # NUM_FBLOCK=NUM_FBLOCK,
            # CBLOCK=CBLOCK,
            # NUM_CBLOCK=NUM_CBLOCK,
        )

    return dq, dk, dv


class LaspFuseParallelAg(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, s, cp_group: dist.ProcessGroup, use_fused_kernel):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        s = s.contiguous()
        # s: (h, 1, 1)
        n = q.shape[-2]
        cp_rank = dist.get_rank(cp_group)
        cp_size = dist.get_world_size(cp_group)

        assert n > BLOCK and n % BLOCK == 0, f"{n} should be divisible by BLOCK"

        # (b, h, NUM_BLOCK + 1, d, e)
        kv = proc_kv(
            q,
            k,
            v,
            s,
            BLOCK,
            CBLOCK,
            cp_size=cp_size,
            use_fused_kernel=use_fused_kernel,
        )

        if cp_size > 1:
            # 假设GPU-1持有的BLOCK编号是3, 4, 5
            # 则local_kv取的是BLOCK 5计算完毕后，由BLOCK 3 ~ 5包含的局部序列的hidden state
            local_kv = kv[:, :, -1].contiguous()
            # 每个设备上都创建一份[cp_size, B, H, K, V]形状的完整hidden states备份
            allgather_kv = torch.empty(
                [cp_size, *local_kv.shape],
                dtype=kv.dtype,
                device=kv.device,
            )
            # 通过all gather操作，将card 0, ..., card n
            dist.all_gather_into_tensor(
                allgather_kv,
                local_kv,
                group=cp_group,
            )
        # PREFIX_SUM(1.0):
        # global_kv_list = list(allgather_kv.chunk(cp_size, dim=0))
        # KV = torch.zeros_like(local_kv).to(dtype=torch.float32)
        # block_decay = torch.exp(-s.to(torch.float32) * n)
        # for i in range(cp_rank):
        #     KV = block_decay * KV + global_kv_list[i]
        if cp_rank > 0:
            # PREFIX_SUM(2.0):
            # block_decay = torch.exp(-s.to(torch.float32) * n)[None, None, :]
            # block_decay = block_decay.repeat_interleave(cp_rank, dim=0)
            # block_decay[0, ...] = 1.0
            # block_decay = torch.cumprod(block_decay, dim=0).flip(dims=[0])
            # PREFIX_SUM(3.0):
            block_decay = rearrange(
                torch.pow(
                    torch.exp(-s.to(torch.float32) * n).unsqueeze(-1),
                    torch.arange(cp_rank - 1, -1, -1, device=s.device),
                ).unsqueeze(0),
                "... i -> i ...",
            )
            decay_kv = block_decay * allgather_kv[:cp_rank]
            KV = torch.sum(decay_kv, dim=0)
        else:
            # case: cp_rank == 0 or cp_size == 1
            local_kv_shape = kv[:, :, -1].shape
            KV = torch.zeros(local_kv_shape, device=kv.device)

        o, KV = lasp_forward(q, k, v, s, kv, KV, BLOCK, CBLOCK)

        ctx.save_for_backward(q, k, v, s, kv, KV)
        ctx.cp_group = cp_group
        ctx.use_fused_kernel = use_fused_kernel

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, s, kv, KV = ctx.saved_tensors
        do = do.contiguous()

        use_fused_kernel = ctx.use_fused_kernel

        cp_group = ctx.cp_group
        cp_size = dist.get_world_size(cp_group)
        cp_rank = dist.get_rank(cp_group)

        b, h, n, d = q.shape
        e = v.shape[-1]

        # (b, h, NUM_BLOCK + 1, d, e)
        dkv = proc_dkv(q, k, v, do, s, BLOCK, CBLOCK, cp_size, use_fused_kernel)

        if cp_size > 1:
            local_dkv = dkv[:, :, -1].contiguous()
            allgather_dkv = torch.empty(
                [cp_size, *local_dkv.shape],
                dtype=dkv.dtype,
                device=dkv.device,
            )

            dist.all_gather_into_tensor(
                allgather_dkv,
                local_dkv,
                group=cp_group,
            )

        # update DKV
        if cp_rank < cp_size - 1:
            # PREFIX_SUM(3.0):
            block_decay = rearrange(
                torch.pow(
                    torch.exp(-s.to(torch.float32) * n).unsqueeze(-1),
                    torch.arange(cp_size - cp_rank - 1, device=s.device),
                ).unsqueeze(0),
                "... i -> i ...",
            )
            decay_dkv = block_decay * allgather_dkv[cp_rank + 1 :]
            DKV = torch.sum(decay_dkv, dim=0)
        else:
            # case: cp_rank == cp_size - 1 or cp_size == 1
            local_dkv_shape = dkv[:, :, -1].shape
            DKV = torch.zeros(local_dkv_shape, device=dkv.device)

        dq, dk, dv = lasp_backward(
            q, k, v, s, do, kv, KV, dkv, DKV, BLOCK, CBLOCK
        )

        return dq, dk, dv, None, None, None


def lasp_fuse_parallel_pretune_all_gather(
    q, k, v, ed, cp_group: dist.ProcessGroup, use_fused_kernel: bool
):
    b, h, n, d = q.shape
    e = v.shape[-1]

    if d >= 128:
        m = 128
    else:
        m = 64
    arr = [m * i for i in range(d // m + 1)]
    if arr[-1] != d:
        arr.append(d)
    n = len(arr)
    output = 0
    for i in range(n - 1):
        s = arr[i]
        e = arr[i + 1]
        q1 = q[..., s:e]
        k1 = k[..., s:e]
        o = LaspFuseParallelAg.apply(q1, k1, v, ed, cp_group, use_fused_kernel)
        output = output + o
    return output
