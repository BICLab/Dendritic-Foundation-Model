import torch
import torch.nn as nn
import torch.distributed as dist

import triton
import triton.language as tl

from ._gla_triton_pretune_impl import (
    prepare_chunk_indices, 
    chunk_gla_fwd, chunk_gla_bwd, 
)
from typing import Optional

from ._gla_utils import input_guard

class GLAGlobalFunc(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx, 
        q: torch.Tensor, # [B, H, T_device, K]
        k: torch.Tensor, # [B, H, T_device, K]
        v: torch.Tensor, # [B, H, T_device, V]
        g: torch.Tensor, # [B, H, T_device, K]
        cp_group: dist.ProcessGroup, 
        scale: Optional[int] = 1,
        initial_state: torch.Tensor = None,
        output_final_state: bool = True,
        cu_seqlens: Optional[torch.LongTensor] = None,
        head_first: bool = True, 
        use_fused_kernel: bool = True, 
    ):
        """
        1. 各显卡分别调用chunk_gla，计算该显卡持有序列对应的局部线性注意力输出
        [注]
        (1)下列代码逻辑保持了chunk_gla的核心计算逻辑
        (2)反向传播参数保存移动至第6部分起始位置
        """
        T = q.shape[2] if head_first else q.shape[1]
        chunk_size = min(64, max(16, triton.next_power_of_2(T)))
        offsets = None

        # 2-d indices denoting the offsets of chunks in each sequence
        # for example, if the passed `offsets` is [0, 100, 356] and `chunk_size` is 64,
        # then there are 2 and 4 chunks in the 1st and 2nd sequences respectively, and `indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        indices = prepare_chunk_indices(offsets, chunk_size) if offsets is not None else None
        g_cumsum, A, h, ht, o_intra_device = chunk_gla_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            g_cumsum=None,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            offsets=offsets,
            indices=indices,
            head_first=head_first,
            chunk_size=chunk_size
        )

        device_final_state = ht.clone()

        """
        2. 针对各显卡的final state作all gather操作，令每个设备都能完整访问所有device level hidden state checkpoints
        final_state: [B, H, K, V]
        """
        b, h, n, d = q.shape
        cp_rank = dist.get_rank(cp_group)
        cp_size = dist.get_world_size(cp_group)

        device_final_state = device_final_state.contiguous() # [B, H, K, V]
        if cp_size > 1:
            device_final_state_allgather = torch.empty(
                [cp_size, *device_final_state.shape], # [cp_size, B, H, K, V]
                dtype=device_final_state.dtype, 
                device=device_final_state.device, 
            )
            torch.cuda.synchronize()
            dist.all_gather_into_tensor(
                device_final_state_allgather,
                device_final_state,
                group=cp_group, 
            )
        else:
            device_final_state_allgather = device_final_state

        """
        3.汇聚各分块级别cumsum of log decay
        [B, H, T_Device, K] -> [B, H, K]
        O = (Q * Lambda) @ (K / Lambda).transpose() @ V
        设备内注意力：直接调用chunk_gla完成
        设备间注意力：需要(Q * Lambda) @ (K / Lambda)_inter_device
        (1) 设备内部的cumsum of log decay -> 逐元素应用于设备内部的query
        (2) 设备内部inverse of cumsum log decay -> 逐元素应用于设备内部的K
        (3) block decay: 整个设备内部的cumsum of log decay，应用于H_{cp_rank - 1}
        """

        # 3.1 设备内部的cumsum of log decay, inverse cumsum of log decay
        # Compute cumulative decay explicitly; this path does not depend on
        # an external GLA cumulative-decay helper.
        g = g.to(dtype=torch.float32)
        device_g_cumsum = torch.cumsum(
            g, dim=2
        ) # [B, H, T_device, K] -> [B, H, T_device, K]
        device_g_sum = device_g_cumsum[:, :, -1, :] # [B, H, K]
        # [当前时间步, ..., 最后一个时间步]的cumsum of log decay
        device_g_cumsum_inverse = (
            device_g_sum[:, :, None, :] - device_g_cumsum + g
        )

        # 3.2 设置device_g_sum的buffer，利用all gather操作，获取所有设备级local cumsum of log decay
        device_g_sum = device_g_sum.contiguous()
        device_g_prod = torch.exp(device_g_sum).contiguous()
        if cp_size > 1:
            # 3.3 汇聚所有device level local cumsum of log decay
            device_g_prod_allgather = torch.empty(
                [cp_size, *device_g_prod.shape], 
                device = device_g_prod.device, 
                dtype = device_g_prod.dtype, 
            )
            torch.cuda.synchronize()
            dist.all_gather_into_tensor(
                device_g_prod_allgather, # [cp_size, B, H, K]
                device_g_prod, 
                group=cp_group, 
            )
        else:
            device_g_prod_allgather = device_g_prod[None, ...]
        """
        4.根据各分块的cumsum of log decay, 及device level局部注意力输出，算global hidden state
        每个device负责其对应索引(cp_rank)的global final state计算
        """
        global_final_state = torch.zeros_like(device_final_state)
        if cp_rank > 0:
            for i in range(cp_rank): # [0, 1, ..., cp_rank - 1]
                # global_final_state: 
                # S_i表示计算完0, 1, ..., i设备为止，hidden state的最新状态
                global_final_state = (
                    (
                        device_g_prod_allgather[i][:, :, :, None] * global_final_state # [B, H, K, 1] * [B, H, K, V]
                    ) + 
                    (
                        device_final_state_allgather[i] # (K_{[t]} / A_{[t]}) @ V_{[t]}.transpose()
                    )
                )
        else:
            pass

        """
        5.计算各设备级分块的全局注意力输出
        (1)设备级别chunk_gla计算的是((Q * A)(K / A).transpose() * M) @ V  
        (2)需补充计算(Q * A) @ S_{[t - 1]}  
        """
        device_g_cumprod = torch.exp(device_g_cumsum)
        device_g_cumprod_inverse = torch.exp(device_g_cumsum_inverse)
        q_bar = q * device_g_cumprod # [B, H, T_device, K] * [B, H, T_device, K]
        o_inter_device = q_bar @ global_final_state # [B, H, T, K] @ [B, H, K, V] -> [B, H, T, V]
        o = o_intra_device + o_inter_device
        
        """
        6.保存必要信息用于反向传播
        (1)chunk_gla前向传播保存的张量
            (i)q/k/v/g：输入张量
            (ii)g_cumsum：局部累积和
            (iii)initial_state：输入初始状态
            (iv)A: 与decay有关的中间状态
        (2)global_final_state: 参与dQ跨设备部分计算
        (3)device_g_prod_allgather: 用于dS_{i + 1} -> dS_{i}的设备间更新
        (4)device_g_cumprod: 用于重计算（Q_{[i]} * Lambda_{[i]}）
        (5)device_g_cumprod_inverse: 用于重计算dv
        """
        # recompute g_cumsum in bwd pass
        if g.dtype != torch.float:
            g_cumsum = None
        else:
            g = None
        # (1)chunk_gla前向传播保存的张量
        ctx.save_for_backward(
            # (1)chunk_gla前向传播保存的张量
            q, k, v, g, g_cumsum, initial_state, A, 
            # (2)global_final_state: 参与dQ跨设备部分计算
            global_final_state, 
            # (3)device_g_prod_allgather: 用于dS_{i + 1} -> dS_{i}的设备间更新
            device_g_prod_allgather, 
            # (4)device_g_cumprod: 用于重计算（Q_{[i]} * Lambda_{[i]}）
            device_g_cumprod, 
            # (5)device_g_cumprod_inverse: 用于重计算dv
            device_g_cumprod_inverse, 
        )
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.offsets = offsets
        ctx.indices = indices
        ctx.head_first = head_first
        # return o, ht
        ctx.cp_group = cp_group # ()context parallel group
        ctx.use_fused_kernel = use_fused_kernel # () use_fused_kernel
        ctx.cp_group = cp_group

        return o

    @staticmethod
    @input_guard
    def backward(
        ctx, 
        do: torch.Tensor,
    ):
        cp_group = ctx.cp_group
        cp_rank = dist.get_rank(group=cp_group)
        cp_size = dist.get_world_size(group=cp_group)
        """
        0.加载前向过程保存的必要张量、元参数
        """

        (
            q, k, v, g, g_cumsum, initial_state, A, 
            global_final_state, 
            device_g_prod_allgather, 
            device_g_cumprod, 
            device_g_cumprod_inverse, 
        ) = ctx.saved_tensors
        chunk_size = ctx.chunk_size 
        scale = ctx.scale
        offsets = ctx.offsets
        indices = ctx.indices
        head_first = ctx.head_first
        
        """
        2.汇聚设备级局部梯度检查点
        """
        batch_size, n_heads, _, key_dim = q.shape
        value_dim = v.shape[-1]
        
        # (1)o_{[i]} = o_{[i], intra} + o_{[i], inter}
        # => do_{[i], intra} == do_{[i]}, do_{[i], inter} = do_{[i]}

        # (2)s_{[i]}参与两类运算：
        #   (i) o_{[i + 1], inter} = q_bar_{[i + 1]} @ s_{[i]}
        #   (ii) s_{[i + 1]} = Lambda_{[i + 1]} @ s_{[i]} + k_tilde_{[i + 1]} @ v_{[i + 1]}.transpose()
        # => ds_{[i]} = ds_{[i], iterate} + ds_{[i], output}
        #   (i)ds_{[i], iterate} = Lambda_{[i + 1]} @ ds_{[i + 1]}
        #   (ii)ds_{[i], output} = q_bar_{[i + 1]}.transpose() @ do_{[i + 1], inter}

        """
        3.准备反向传播所需的中间结果
        """
        k_tilde = k * device_g_cumprod_inverse # [B, H, T, K] * [B, H, T, K] -> [B, H, T, K]
        q_bar = q * device_g_cumprod

        """
        4.计算设备级(Lambda_{[i]} @ Q_i) @ dO_{[i]}.transpose()，并all_gather_into_tensor
        """
        # o_{[i + 1], inter} = q_bar_{[i + 1]} @ s_{[i]}
        # => ds_{[i], output} = q_bar_{[i + 1]}.transpose() @ do_{[i + 1], inter}
        device_q_bar_do = q_bar.transpose(-2, -1) @ do # [B, H, K, T] @ [B, H, T, V] -> [B, H, K, V]
        device_q_bar_do_allgather = torch.empty(
            size=(cp_size, *device_q_bar_do.shape), 
            device=device_q_bar_do.device, dtype=device_q_bar_do.dtype, 
        )
        dist.all_gather_into_tensor(
            output_tensor=device_q_bar_do_allgather,
            input_tensor=device_q_bar_do,
            group=cp_group, 
        )

        """
        5.根据隐藏状态局部梯度，计算隐藏状态全局梯度
        """
        # s_{[i]}参与两类运算：
        #   (i) o_{[i + 1], inter} = q_bar_{[i + 1]} @ s_{[i]}
        #   (ii) s_{[i + 1]} = Lambda_{[i + 1]} * s_{[i]} + k_tilde_{[i + 1]} @ v_{[i + 1]}.transpose()
        # => ds_{[i]} = ds_{[i], iterate} + ds_{[i], output}
        #   (i)ds_{[i], iterate} = Lambda_{[i + 1]} * ds_{[i + 1]}
        #   (ii)ds_{[i], output} = q_bar_{[i + 1]}.transpose() @ do_{[i + 1], inter}

        # 合并形式：ds_{[i]} = Lambda_{[i + 1]} * ds_{[i + 1]} + q_bar_{[i + 1]}.transpose() @ do_{[i + 1], inter}
        ds = torch.zeros(
            size=(batch_size, n_heads, key_dim, value_dim), 
            device=q.device, dtype=q.dtype, 
        )
        if cp_rank < cp_size - 1:
            for i in range(cp_size - 1, cp_rank, -1): # [cp_size - 1, ..., cp_rank + 1]
                ds = (
                    (device_g_prod_allgather[i][..., None] * ds) # Lambda_{[i + 1]} * ds_{[i + 1]}
                    + # [B, H, K, 1] * [B, H, K, V]
                    device_q_bar_do_allgather[i] # q_bar_{[i + 1]}.transpose() @ do_{[i + 1], inter}
                )

        """
        1.各设备独立调用chunk_gla_bwd，分别计算设备级局部反向传播的梯度
        [注]
        (1)调用chunk_gla_bwd，即可实现ChunkGLAFunction的全部计算逻辑
        (2)dh0：表示隐藏状态的设备级局部梯度，需逆序all_gather_into_tensor，以获取隐藏状态的全局梯度
        """
        dq, dk, dv, _, _ = chunk_gla_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            g_cumsum=g_cumsum,
            scale=scale,
            h=None,
            A=A,
            initial_state=global_final_state, # None -> global_final_state
            do=do,
            dht=ds, # None -> ds
            offsets=offsets,
            indices=indices,
            head_first=head_first,
            chunk_size=chunk_size
        )

        # """
        # 6.计算dq_inter
        # """
        
        # # (1)o_{[i], inter} = q_bar_{[i]} @ S_{[i - 1]}
        # # => dq_bar_{[i]} = do_{[i], inter} @ S_{[i - 1]}.transpose()
        # dq_bar_inter = do @ global_final_state.transpose(-2, -1) # [B, H, T, V] @ [B, H, V, K] -> [B, H, T, K]

        # # (2)q_bar_{[i]} = q_{[i]} * Lambda_{[i]}
        # # => dq_{[i]} = dq_bar_{[i]} * Lambda_{[i]}
        # dq_inter = dq_bar_inter * device_g_cumprod # [B, H, T, K] * [B, H, T, K] -> [B, H, T, K]

        # """
        # 7.计算dk_inter
        # """
        # # (1) s_{[i]} = ... + k_tilde_{[i]}.transpose() @ v_{[i]}
        # # => dk_tilde_{[i]} = v_{[i]} @ ds_{[i]}.transpose()
        # dk_tilde_inter = v @ ds.transpose(-2, -1) # [B, H, T, V] @ [B, H, V, K] -> [B, H, T, K]
        # # (2) k_tilde_{[i]} = k_{[i]} * lambda_reverse_{[i]}
        # # => dk_{[i]} = dk_tilde_{[i]} * lambda_reverse_{[i]}
        # dk_inter = dk_tilde_inter * device_g_cumprod

        # """
        # 8.计算dv_inter
        # """
        # # ds_{[i]} = ... + k_tilde_{[i]}.transpose() @ v_{[i]}
        # # => dv_{[i]} = k_tilde_{[i]} @ ds_{[i]}
        # dv_inter = k_tilde @ ds # [B, H, T, K] @ [B, H, K, V] -> [B, H, T, V]

        # """
        # 9.梯度合并
        # """
        # dq = dq_intra + dq_inter
        # dk = dk_intra + dk_inter
        # dv = dv_intra + dv_inter

        return dq, dk, dv, None, None, None, None, None, None, None, None
        
chunk_gla_global = GLAGlobalFunc.apply