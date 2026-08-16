#    Copyright 2024 OpenNLPLab
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# coding=utf-8
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Out,
    S,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_ve,
    stride_oz,
    stride_oh,
    stride_om,
    stride_oe,
    stride_sh,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_DECAY: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_h = off_hz % H
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL_QK)
    offs_e = tl.arange(0, BLOCK_DMODEL_V)
    # get current offset of q k v
    off_q = (off_hz * stride_qh + offs_m[:, None] * stride_qm +
             offs_k[None, :] * stride_qk)
    off_k = (off_hz * stride_kh + offs_n[:, None] * stride_kn +
             offs_k[None, :] * stride_kk)
    off_v = (off_hz * stride_vh + offs_n[:, None] * stride_vn +
             offs_e[None, :] * stride_ve)
    off_o = (off_hz * stride_oh + offs_m[:, None] * stride_om +
             offs_e[None, :] * stride_oe)

    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # initialize pointer to m and l
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    # loop over k, v and update accumulator
    lo = 0
    # print(start_m)
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(
            k_ptrs + start_n * stride_kn,
            mask=(start_n + offs_n)[:, None] < N_CTX,
            other=0.0,
        )
        v = tl.load(
            v_ptrs + start_n * stride_vn,
            mask=(start_n + offs_n)[:, None] < N_CTX,
            other=0.0,
        )
        # -- compute qk ---
        # qk = tl.dot(q, k)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # qk += tl.dot(q, k, trans_b=True)
        qk += tl.dot(q, tl.trans(k))
        if IS_CAUSAL:
            index = offs_m[:, None] - (start_n + offs_n[None, :])
            if USE_DECAY:
                S_block_ptr = S + off_h * stride_sh
                s = tl.load(S_block_ptr)
                s_index = s * index
                s_index = tl.where(s_index >= 0, -s_index, float("-inf"))
                qk = tl.exp(s_index) * qk
            else:
                qk = tl.where(index >= 0, qk, 0)
        acc += tl.dot(qk, v.to(qk.dtype))

    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc.to(q.dtype), mask=offs_m[:, None] < N_CTX)


@triton.jit
def _bwd_kernel_kv(
    Q,
    K,
    V,
    S,
    DO,
    DQ,
    DK,
    DV,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_ve,
    stride_oz,
    stride_oh,
    stride_om,
    stride_oe,
    stride_sh,
    Z,
    H,
    N_CTX,
    num_block,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_DECAY: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_z = off_hz // H
    off_h = off_hz % H
    # offset pointers for batch/head
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_oz + off_h * stride_oh
    DQ += off_z * stride_qz + off_h * stride_qh
    DK += off_z * stride_kz + off_h * stride_kh
    DV += off_z * stride_vz + off_h * stride_vh

    # start of q
    if CAUSAL:
        lo = start_n * BLOCK_M
    else:
        lo = 0
    # initialize row/col offsets
    # seqlence offset
    offs_qm = lo + tl.arange(0, BLOCK_M)
    offs_kvn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # feature offset
    offs_qkk = tl.arange(0, BLOCK_DMODEL_QK)
    offs_ve = tl.arange(0, BLOCK_DMODEL_V)
    # row block index
    offs_m = tl.arange(0, BLOCK_M)
    # initialize pointers to value-like data
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_qkk[None, :] * stride_qk)
    k_ptrs = K + (offs_kvn[:, None] * stride_kn +
                  offs_qkk[None, :] * stride_kk)
    v_ptrs = V + (offs_kvn[:, None] * stride_vn + offs_ve[None, :] * stride_ve)
    do_ptrs = DO + (offs_qm[:, None] * stride_om +
                    offs_ve[None, :] * stride_oe)
    dq_ptrs = DQ + (offs_qm[:, None] * stride_qm +
                    offs_qkk[None, :] * stride_qk)
    # initialize dv amd dk
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL_V], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL_QK], dtype=tl.float32)
    # k and v stay in SRAM throughout
    k = tl.load(k_ptrs, mask=offs_kvn[:, None] < N_CTX, other=0.0)
    v = tl.load(v_ptrs, mask=offs_kvn[:, None] < N_CTX, other=0.0)
    # loop over rows
    for start_m in range(lo, num_block * BLOCK_M, BLOCK_M):
        offs_m_curr = start_m + offs_m
        # load q, k, v, do on-chip
        q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < N_CTX, other=0.0)
        qk = tl.dot(q, tl.trans(k))
        # qk = tl.dot(q, k, trans_b=True)
        if CAUSAL:
            index = offs_m_curr[:, None] - offs_kvn[None, :]
            if USE_DECAY:
                S_block_ptr = S + off_h * stride_sh
                s = tl.load(S_block_ptr)
                s_index = s * index
                s_index = tl.where(s_index >= 0, -s_index, float("-inf"))
                s = tl.exp(s_index)
                qk = qk * s
            else:
                qk = tl.where(index >= 0, qk, 0)

        p = qk
        # compute dv
        do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < N_CTX, other=0.0)
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
        dp = tl.dot(do, tl.trans(v).to(do.dtype))
        if CAUSAL:
            if USE_DECAY:
                dp = dp * s
            else:
                dp = tl.where(index >= 0, dp, 0)

        dk += tl.dot(tl.trans(dp.to(q.dtype)), q).to(tl.float32)

        # increment pointers
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_om
    # write-back
    dv_ptrs = DV + (offs_kvn[:, None] * stride_vn +
                    offs_ve[None, :] * stride_ve)
    dk_ptrs = DK + (offs_kvn[:, None] * stride_kn +
                    offs_qkk[None, :] * stride_kk)
    tl.store(dv_ptrs, dv, mask=offs_kvn[:, None] < N_CTX)
    tl.store(dk_ptrs, dk, mask=offs_kvn[:, None] < N_CTX)


@triton.jit
def _bwd_kernel_q(
    Q,
    K,
    V,
    S,
    DO,
    DQ,
    DK,
    DV,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_ve,
    stride_oz,
    stride_oh,
    stride_om,
    stride_oe,
    stride_sh,
    Z,
    H,
    N_CTX,
    num_block,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_DECAY: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    # offset pointers for batch/head
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_oz + off_h * stride_oh
    DQ += off_z * stride_qz + off_h * stride_qh
    # feature offset
    offs_qkk = tl.arange(0, BLOCK_DMODEL_QK)
    offs_ve = tl.arange(0, BLOCK_DMODEL_V)
    # row block index
    offs_m = tl.arange(0, BLOCK_M)
    # row block index
    offs_qm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # do
    do_ptrs = DO + (offs_qm[:, None] * stride_om +
                    offs_ve[None, :] * stride_oe)
    dq_ptrs = DQ + (offs_qm[:, None] * stride_qm +
                    offs_qkk[None, :] * stride_qk)

    do = tl.load(do_ptrs, mask=offs_qm[:, None] < N_CTX, other=0.0)

    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL_QK], dtype=tl.float32)
    lo = 0
    hi = (start_m + 1) * BLOCK_M if CAUSAL else N_CTX

    offs_m_curr = start_m * BLOCK_M + offs_m

    for start_n in range(0, num_block):
        offs_kvn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k_ptrs = K + (offs_kvn[:, None] * stride_kn +
                      offs_qkk[None, :] * stride_kk)
        v_ptrs = V + (offs_kvn[:, None] * stride_vn +
                      offs_ve[None, :] * stride_ve)
        # k and v stay in SRAM throughout
        k = tl.load(k_ptrs, mask=offs_kvn[:, None] < N_CTX, other=0.0)
        v = tl.load(v_ptrs, mask=offs_kvn[:, None] < N_CTX, other=0.0)
        # dp = do vT
        dp = tl.dot(do, tl.trans(v).to(do.dtype))
        if CAUSAL:
            index = offs_m_curr[:, None] - offs_kvn[None, :]
            if USE_DECAY:
                S_block_ptr = S + off_h * stride_sh
                s = tl.load(S_block_ptr)
                s_index = s * index
                s_index = tl.where(s_index >= 0, -s_index, float("-inf"))
                s = tl.exp(s_index)
                dp = dp * s
            else:
                dp = tl.where(index >= 0, dp, 0)
        # dq = dq + dp k
        dq += tl.dot(dp.to(k.dtype), k)

    tl.store(dq_ptrs, dq, mask=offs_qm[:, None] < N_CTX)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, s):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        s = s.contiguous()
        # only support for Ampere now
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError(
                "Lightning attention currently only supported for compute capability >= 80"
            )
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        # right
        o = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2], v.shape[-1]),
            dtype=q.dtype,
            device=q.device,
        )

        BLOCK_M = 128
        BLOCK_N = 64
        num_warps = 4 if Lk <= 64 else 8
        num_stages = 1

        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
        use_decay = s.shape[0] > 0
        _fwd_kernel[grid](
            q,
            k,
            v,
            o,
            s,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            s.stride(0),
            q.shape[0],
            q.shape[1],
            q.shape[2],
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL_QK=Lk,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL_V=Lv,
            IS_CAUSAL=causal,
            USE_DECAY=use_decay,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        ctx.save_for_backward(q, k, v, s)
        ctx.grid = grid
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_DMODEL_QK = Lk
        ctx.BLOCK_N = BLOCK_N
        ctx.BLOCK_DMODEL_V = Lv
        ctx.causal = causal
        ctx.use_decay = use_decay
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, s = ctx.saved_tensors
        BLOCK_M = 32
        BLOCK_N = 32
        num_warps = 4
        num_stages = 1

        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        grid_kv = (triton.cdiv(k.shape[2],
                               BLOCK_N), k.shape[0] * k.shape[1], 1)
        _bwd_kernel_kv[grid_kv](
            q,
            k,
            v,
            s,
            do,
            dq,
            dk,
            dv,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            do.stride(3),
            s.stride(0),
            q.shape[0],
            q.shape[1],
            q.shape[2],
            grid_kv[0],
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL_QK=ctx.BLOCK_DMODEL_QK,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL_V=ctx.BLOCK_DMODEL_V,
            CAUSAL=ctx.causal,
            USE_DECAY=ctx.use_decay,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        grid_q = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)

        _bwd_kernel_q[grid_q](
            q,
            k,
            v,
            s,
            do,
            dq,
            dk,
            dv,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            do.stride(3),
            s.stride(0),
            q.shape[0],
            q.shape[1],
            q.shape[2],
            grid_q[0],
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL_QK=ctx.BLOCK_DMODEL_QK,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL_V=ctx.BLOCK_DMODEL_V,
            CAUSAL=ctx.causal,
            USE_DECAY=ctx.use_decay,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        return dq.to(q.dtype), dk, dv, None, None


attention = _attention.apply


def lightning_attention(q, k, v, causal, ed):
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
    for i in range(n - 1):
        s = arr[i]
        e = arr[i + 1]
        q1 = q[..., s:e]
        k1 = k[..., s:e]
        o = attention(q1, k1, v, causal, ed)
        output = output + o

    return output
