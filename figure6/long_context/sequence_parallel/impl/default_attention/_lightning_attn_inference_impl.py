import torch
import triton
import triton.language as tl

# 16 clear


@triton.jit
def _fwd_diag_kernel(
    Q,
    K,
    V,
    Out,
    S,
    b: tl.constexpr,
    h: tl.constexpr,
    n,
    d: tl.constexpr,
    e: tl.constexpr,
    # BLOCK_DMODEL_QK: tl.constexpr, BLOCK_DMODEL_V: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
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

    q = tl.load(
        Q_block_ptr, mask=block_offset + q_index[:, None] < n, other=0.0
    ).to(tl.float32)

    qkv = tl.zeros([CBLOCK, e], dtype=tl.float32)
    # none diag

    for j in range(i + 1):
        kv_index = tl.arange(0, CBLOCK) + j * CBLOCK
        diff = q_index[:, None] - kv_index[None, :]
        s_index = s * diff
        s_index = tl.where(diff >= 0, -s_index, float("-inf"))
        decay = tl.exp(s_index)

        k_trans = tl.load(
            K_trans_block_ptr,
            mask=block_offset + kv_index[None, :] < n,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V_block_ptr,
            mask=block_offset + kv_index[:, None] < n,
            other=0.0,
        ).to(tl.float32)

        qk = tl.dot(q, k_trans) * decay

        qkv += tl.dot(qk, v)

        K_trans_block_ptr += CBLOCK * d
        V_block_ptr += CBLOCK * e

    tl.store(
        O_block_ptr,
        qkv.to(O_block_ptr.dtype.element_ty),
        mask=block_offset + q_index[:, None] < n,
    )


@triton.jit
def _fwd_kv_parallel(
    K,
    V,
    K_decay,
    KV,
    b: tl.constexpr,
    h: tl.constexpr,
    n,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK,
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

    k_block_offset = block_offset * d
    v_block_offset = block_offset * e
    kv_block_offset = off_block * d * e

    k_offset = off_bh * n * d
    v_offset = off_bh * n * e
    kv_offset = off_bh * NUM_BLOCK * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK

    # (CBLOCK, FBLOCK)
    K_trans_block_ptr = (
        K
        + k_offset
        + k_block_offset
        + tl.arange(0, CBLOCK)[None, :] * d  # d x c
        + tl.arange(0, D_FBLOCK)[:, None]
    )
    V_block_ptr = (
        V
        + v_offset
        + v_block_offset
        + tl.arange(0, CBLOCK)[:, None] * e  # c x d
        + tl.arange(0, E_FBLOCK)[None, :]
    )
    KV_block_ptr = (
        KV
        + kv_offset
        + kv_block_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    # s_ptrs = S + off_h
    # s = tl.load(s_ptrs)
    k_decay_ptr = K_decay + off_h * BLOCK + tl.arange(0, CBLOCK)[None, :]

    # compute block array
    kv_index = tl.arange(0, CBLOCK)

    # c_array = tl.arange(0, CBLOCK) + 1
    kv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)

    if off_block == NUM_BLOCK - 1:
        split_n = n - (NUM_BLOCK - 1) * BLOCK
    else:
        split_n = BLOCK
    left_shift = tl.cdiv(split_n, CBLOCK) * CBLOCK - split_n
    num_blocks = min(tl.cdiv(split_n, CBLOCK), NUM_CBLOCK)
    k_decay_ptr += (NUM_CBLOCK - num_blocks) * CBLOCK
    for j in range(num_blocks):
        # right align k, v with CBLOCK
        left_bound = (1 - j) * left_shift
        k_trans = tl.load(
            K_trans_block_ptr - left_shift * d,
            mask=kv_index[None, :] >= left_bound,
            other=0.0,
        )
        v = tl.load(
            V_block_ptr - left_shift * d,
            mask=kv_index[:, None] >= left_bound,
            other=0.0,
        )

        # k_decay = tl.exp(
        #     -s.to(tl.float32) * (BLOCK - ((NUM_CBLOCK-num_blocks+j) * CBLOCK + c_array[None, :]))
        # )
        k_decay = tl.load(k_decay_ptr)
        kv += tl.dot(k_trans * k_decay, v)

        K_trans_block_ptr += CBLOCK * d
        V_block_ptr += CBLOCK * e
        k_decay_ptr += CBLOCK

    tl.store(KV_block_ptr, kv.to(KV_block_ptr.dtype.element_ty))


@triton.jit
def _fwd_kv_reduce(
    K,
    V,
    S,
    KV,
    KV_HISTORY,
    b: tl.constexpr,
    h: tl.constexpr,
    n,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK,
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_h = off_bh % h
    off_d = tl.program_id(1)
    off_e = tl.program_id(2)

    kv_offset = off_bh * NUM_BLOCK * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK

    # (CBLOCK, FBLOCK)
    KV_block_ptr = (
        KV
        + kv_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    s_ptrs = S + off_h
    s = tl.load(s_ptrs)

    # block_decay = tl.exp(-s.to(tl.float32) * BLOCK)

    # Initialize kv from KV_HISTORY
    kv_history_offset = off_bh * d * e
    KV_HISTORY_block_ptr = (
        KV_HISTORY
        + kv_history_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )
    # compute block array
    # last step
    kv_pre = tl.load(KV_HISTORY_block_ptr).to(tl.float32)
    for i in range(NUM_BLOCK):
        block_size = min(n - i * BLOCK, BLOCK)
        block_decay = tl.exp(-s.to(tl.float32) * block_size)

        kv_cur = tl.load(KV_block_ptr).to(tl.float32)
        tl.store(KV_block_ptr, kv_pre.to(KV_block_ptr.dtype.element_ty))

        kv_pre = block_decay * kv_pre + kv_cur
        KV_block_ptr += d * e
    tl.store(KV_HISTORY_block_ptr, kv_pre)


# total parallel
# @triton.jit
@triton.jit
def _fwd_none_diag_kernel(
    Q,
    K,
    V,
    Out,
    S,
    KV,
    b: tl.constexpr,
    h: tl.constexpr,
    n,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK,
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_h = off_bh % h

    off_nc = tl.program_id(1)
    off_n = off_nc // NUM_CBLOCK
    off_c = off_nc % NUM_CBLOCK
    off_e = tl.program_id(2)

    n_offset = off_n * BLOCK
    c_offset = off_c * CBLOCK
    e_offset = off_e * E_FBLOCK
    block_offset = n_offset + c_offset

    q_offset = off_bh * n * d + (n_offset + c_offset) * d
    o_offset = off_bh * n * e + (n_offset + c_offset) * e + e_offset

    # kv: (b, h, NUM_BLOCK + 1, d, e)
    kv_offset = off_bh * NUM_BLOCK * d * e + off_n * d * e + e_offset

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
    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    c_array = tl.arange(0, CBLOCK)

    kv = tl.load(KV_block_ptr).to(tl.float32)
    q_index = block_offset + tl.arange(0, CBLOCK)
    # q = tl.load(Q_block_ptr).to(tl.float32)
    # BUG: oob
    q = tl.load(Q_block_ptr, mask=q_index[:, None] < n, other=0.0).to(
        tl.float32
    )

    q_decay = tl.exp(-s.to(tl.float32) * (off_c * CBLOCK + c_array[:, None]))
    qkv_none_diag = tl.dot(q, kv) * q_decay

    # qkv_diag = tl.load(O_block_ptr).to(tl.float32)
    # BUG: oob
    qkv_diag = tl.load(O_block_ptr, mask=q_index[:, None] < n, other=0.0).to(
        tl.float32
    )

    qkv = qkv_diag + qkv_none_diag

    # tl.store(O_block_ptr, qkv.to(O_block_ptr.dtype.element_ty))
    # BUG: oob
    tl.store(
        O_block_ptr,
        qkv.to(O_block_ptr.dtype.element_ty),
        mask=q_index[:, None] < n,
    )


# bwd
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
    NUM_CBLOCK: tl.constexpr,
):
    off = tl.program_id(0)
    off_bh = off // NUM_BLOCK
    off_block = off % NUM_BLOCK
    off_cblock = tl.program_id(1)

    off_h = off_bh % h

    #####
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
    cblock_offset_1d = tl.arange(0, CBLOCK)

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

    do_index_mask = (block_offset + cblock_offset + cblock_offset_1d)[
        :, None
    ] < n
    do = tl.load(DO_block_ptr, mask=do_index_mask, other=0.0).to(tl.float32)
    dq = tl.zeros([CBLOCK, d], dtype=tl.float32)

    i = off_cblock
    do_index = tl.arange(0, CBLOCK) + i * CBLOCK
    for j in range(i + 1):
        v_index = tl.arange(0, CBLOCK) + j * CBLOCK

        k = tl.load(
            K_block_ptr, mask=block_offset + v_index[:, None] < n, other=0.0
        ).to(tl.float32)
        v_trans = tl.load(
            V_trans_block_ptr,
            mask=block_offset + v_index[None, :] < n,
            other=0.0,
        ).to(tl.float32)

        # compute
        diff = do_index[:, None] - v_index[None, :]
        s_index = s * diff
        s_index = tl.where(diff >= 0, -s_index, float("-inf"))
        diag_decay = tl.exp(s_index)

        dqk = tl.dot(do, v_trans) * diag_decay
        dq += tl.dot(dqk, k)

        K_block_ptr += CBLOCK * d
        V_trans_block_ptr += CBLOCK * e

    tl.store(
        DQ_block_ptr,
        dq.to(DQ_block_ptr.dtype.element_ty),
        mask=do_index_mask,
    )

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

    v_trans_index_mask = (block_offset + cblock_offset + cblock_offset_1d)[
        None, :
    ] < n
    v_trans = tl.load(V_trans_block_ptr, mask=v_trans_index_mask, other=0.0).to(
        tl.float32
    )
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
    k_index_mask = (block_offset + cblock_offset + cblock_offset_1d)[
        :, None
    ] < n
    k = tl.load(K_block_ptr, mask=k_index_mask, other=0.0).to(tl.float32)
    for j in range(i, NUM_CBLOCK):
        do_index = tl.arange(0, CBLOCK) + j * CBLOCK

        q_trans = tl.load(
            Q_trans_block_ptr,
            mask=block_offset + do_index[None, :] < n,
            other=0.0,
        ).to(tl.float32)
        do = tl.load(
            DO_block_ptr, mask=block_offset + do_index[:, None] < n, other=0.0
        ).to(tl.float32)

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
        DK_trans_block_ptr,
        dk_trans.to(DK_trans_block_ptr.dtype.element_ty),
        mask=v_trans_index_mask,
    )
    tl.store(
        DV_block_ptr,
        dv.to(DV_block_ptr.dtype.element_ty),
        mask=k_index_mask,
    )


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
    kv_offset = off_bh * NUM_BLOCK * d * e
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
    do_index = block_offset + tl.arange(0, CBLOCK)
    for j in range(NUM_CBLOCK):
        do = tl.load(DO_block_ptr, mask=do_index[:, None] < n, other=0.0).to(
            tl.float32
        )
        q_trans = tl.load(
            Q_trans_block_ptr, mask=do_index[None, :] < n, other=0.0
        ).to(tl.float32)
        q_decay_trans = tl.exp(
            -s.to(tl.float32) * (j * CBLOCK + c_array[None, :])
        )
        dkv += tl.dot(q_trans * q_decay_trans, do)

        DO_block_ptr += CBLOCK * e
        Q_trans_block_ptr += CBLOCK * d
        do_index += CBLOCK

    tl.store(DKV_block_ptr, dkv.to(DKV_block_ptr.dtype.element_ty))


# @triton.jit
@triton.jit
def _bwd_dkv_reduce(
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
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    start_m = 0
    off_bh = tl.program_id(0)
    off_h = off_bh % h
    off_d = tl.program_id(1)
    off_e = tl.program_id(2)

    kv_offset = off_bh * NUM_BLOCK * d * e
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
        tl.store(DKV_block_ptr, dkv.to(DKV_block_ptr.dtype.element_ty))

        dkv = block_decay * dkv + dkv_current


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
    off_h = off_bh % h

    off_nc = tl.program_id(1)
    off_n = off_nc // NUM_CBLOCK
    off_c = off_nc % NUM_CBLOCK
    off_de = tl.program_id(2)

    n_offset = off_n * BLOCK
    c_offset = off_c * CBLOCK
    d_offset = off_de * D_FBLOCK
    e_offset = off_de * E_FBLOCK
    block_offset = n_offset + c_offset
    cblock_offset_1d = tl.arange(0, CBLOCK)

    qk_offset = off_bh * n * d + (n_offset + c_offset) * d
    v_offset = off_bh * n * e + (n_offset + c_offset) * e
    o_offset = off_bh * n * e + (n_offset + c_offset) * e

    kv_offset = off_bh * NUM_BLOCK * d * e + off_n * d * e
    kv_trans_offset = off_bh * NUM_BLOCK * d * e + off_n * d * e

    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    # dq
    DO_block_ptr = (
        DO
        + o_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )
    KV_trans_block_ptr = (
        KV
        + kv_trans_offset
        + d_offset * e
        + tl.arange(0, D_FBLOCK)[None, :] * e
        + tl.arange(0, e)[:, None]
    )
    DQ_block_ptr = (
        DQ
        + qk_offset
        + d_offset
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, D_FBLOCK)[None, :]
    )

    do_index_mask = (block_offset + cblock_offset_1d)[:, None] < n
    kv_trans = tl.load(KV_trans_block_ptr).to(tl.float32)
    q_decay = tl.exp(
        -s.to(tl.float32) * (off_c * CBLOCK + cblock_offset_1d[:, None])
    )
    do = tl.load(DO_block_ptr, mask=do_index_mask, other=0.0).to(tl.float32)
    dq_none_diag = tl.dot(do, kv_trans) * q_decay
    dq = dq_none_diag + tl.load(DQ_block_ptr, mask=do_index_mask, other=0.0).to(
        tl.float32
    )
    tl.store(
        DQ_block_ptr,
        dq.to(DQ_block_ptr.dtype.element_ty),
        mask=do_index_mask,
    )

    # dk
    DK_trans_block_ptr = (
        DK
        + qk_offset
        + d_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, D_FBLOCK)[:, None]
    )
    DKV_block_ptr = (
        DKV
        + kv_offset
        + d_offset * e
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )
    V_trans_block_ptr = (
        V
        + v_offset
        + tl.arange(0, CBLOCK)[None, :] * e
        + tl.arange(0, e)[:, None]
    )

    v_trans_index_mask = tl.trans(do_index_mask)
    v_trans = tl.load(V_trans_block_ptr, mask=v_trans_index_mask, other=0.0).to(
        tl.float32
    )
    dkv = tl.load(DKV_block_ptr).to(tl.float32)
    k_decay_trans = tl.exp(
        -s.to(tl.float32)
        * (BLOCK - (off_c * CBLOCK + cblock_offset_1d[None, :]))
    )

    dk_none_diag_trans = tl.dot(dkv, v_trans) * k_decay_trans
    dk_trans = dk_none_diag_trans + tl.load(
        DK_trans_block_ptr, mask=v_trans_index_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        DK_trans_block_ptr,
        dk_trans.to(DK_trans_block_ptr.dtype.element_ty),
        mask=v_trans_index_mask,
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
        -s.to(tl.float32)
        * (BLOCK - (off_c * CBLOCK + cblock_offset_1d[:, None]))
    )

    k_index_mask = do_index_mask
    k = tl.load(K_block_ptr, mask=k_index_mask, other=0.0).to(tl.float32)
    dkv_ = tl.load(DKV_block_ptr_).to(tl.float32)
    dv_none_diag = tl.dot(k, dkv_) * k_decay
    dv = dv_none_diag + tl.load(DV_block_ptr, mask=k_index_mask, other=0.0).to(
        tl.float32
    )
    tl.store(
        DV_block_ptr,
        dv.to(DV_block_ptr.dtype.element_ty),
        mask=k_index_mask,
    )


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, s, kv_history):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        s = s.contiguous()
        # only support for Ampere now
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError(
                "Flash attention currently only supported for compute capability >= 80"
            )
        # shape constraints
        b, h, n, d = q.shape
        e = v.shape[-1]
        # right
        o = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)

        BLOCK = 256
        # NUM_BLOCK = q.shape[2] // BLOCK
        # BUG: oob
        NUM_BLOCK = triton.cdiv(n, BLOCK)

        CBLOCK = 64
        # 修改1: CBLOCK 64->32
        CBLOCK = 32
        NUM_CBLOCK = BLOCK // CBLOCK
        assert BLOCK % CBLOCK == 0, "BLOCK must be a multiple of CBLOCK"

        # k_decay.shape = (h, 1, BLOCK)
        array = torch.arange(0, BLOCK, device=q.device) + 1
        # q_decay = torch.exp(-s * array.reshape(-1, 1))
        k_decay = torch.exp(-s * (BLOCK - array.reshape(1, -1)))

        grid = (b * h * NUM_BLOCK, NUM_CBLOCK)
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
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )

        NUM_FBLOCK = 1
        D_FBLOCK = d // NUM_FBLOCK
        assert d % NUM_FBLOCK == 0
        E_FBLOCK = e // NUM_FBLOCK
        assert e % NUM_FBLOCK == 0

        # 修改2: _fwd_kv_parallel 保持不变
        CBLOCK = 64
        NUM_CBLOCK = BLOCK // CBLOCK
        assert BLOCK % CBLOCK == 0, "BLOCK must be a multiple of CBLOCK"

        kv = torch.empty(
            (b, h, NUM_BLOCK, d, e), dtype=torch.float32, device=q.device
        )
        grid = (b * h, NUM_BLOCK, NUM_FBLOCK * NUM_FBLOCK)
        _fwd_kv_parallel[grid](
            k,
            v,
            k_decay,
            kv,
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

        grid = (b * h, NUM_FBLOCK, NUM_FBLOCK)
        _fwd_kv_reduce[grid](
            k,
            v,
            s,
            kv,
            kv_history,
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

        grid = (b * h, NUM_BLOCK * NUM_CBLOCK, NUM_FBLOCK)
        _fwd_none_diag_kernel[grid](
            q,
            k,
            v,
            o,
            s,
            kv,
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

        return o, torch.cat([kv, kv_history.unsqueeze(2)], dim=2)

    @staticmethod
    def backward(ctx, do):
        pass


lightning_attention_ = _attention.apply


def lightning_attention(q, k, v, ed, block_size=256, kv_history=None):
    d = q.shape[-1]
    e = v.shape[-1]
    # arr = f(d)
    if d >= 128:
        m = 128
    else:
        m = 64
    arr = [m * i for i in range(d // m + 1)]
    if arr[-1] != d:
        arr.append(d)
    n = len(arr)
    output = 0
    if kv_history is None:
        # [b, nh, d, e]
        kv_history = torch.zeros(
            (q.shape[0], q.shape[1], d, e), dtype=torch.float32, device=q.device
        )
    else:
        # make sure run in functional programming style
        kv_history = kv_history.clone().contiguous()

    for i in range(n - 1):
        s = arr[i]
        e = arr[i + 1]
        q1 = q[..., s:e]  # .contiguous()
        k1 = k[..., s:e]  # .contiguous()
        # print(output.shape)
        o, kv = lightning_attention_(q1, k1, v, ed, kv_history)
        output = output + o
    return output, kv


def lightning_attention_inference(q, k, v, ed, block_size=256, kv_history=None):
    return lightning_attention(q, k, v, ed, block_size, kv_history)
