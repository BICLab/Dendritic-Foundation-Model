from dataclasses import dataclass
import os

import torch
import torch.distributed
import torch.nn.functional as F
from einops import rearrange

from meepo.module._base_module import MeepoModule, MeepoModuleBaseConfig
from meepo.module.utils import MeepoModuleSpec
from meepo.utils.inference_cache_mgr import CacheType, inference_cache_mgr

# from ._lasp_triton_pretune_impl import (
#     lasp_fuse_parallel_pretune_all_gather,
# )
from ._lightning_attn_inference_impl import lightning_attention_inference
from ._lightning_attn_triton_impl import (
    context_parallel_lightning_attention,
)

from ._gla_triton_pretune_impl import (
    chunk_gla, 
)

from ._gla_global_func import (
    chunk_gla_global, 
)
from ._gla_inference_impl import fused_recurrent_gla

@dataclass
class DefaultLinearAttentionConfig(MeepoModuleBaseConfig):
    overlap: bool = False
    """
    
    """
    use_lasp: bool = True
    use_lasp_fused_kernel: bool = False
    is_sparsela: bool = True
    sparse_rate: float = float(os.getenv("SP_RATE", "0.5"))

    def validate(self):
        assert self.overlap is False, "overlap is not supported Currently"
        if not 0.0 <= self.sparse_rate <= 1.0:
            raise ValueError("sparse_rate must be between 0.0 and 1.0")
        if self.use_lasp_fused_kernel:
            assert (
                self.use_lasp
            ), "can not use_lasp_fused_kernel without use_lasp "

        super().validate()


class DefaultLinearAttention(MeepoModule):
    _is_base_module = True

    def __init__(
        self,
        config: DefaultLinearAttentionConfig,
    ):
        super().__init__(config)
        assert isinstance(config, DefaultLinearAttentionConfig)
        self.config: DefaultLinearAttentionConfig

        if self.config.use_lasp_fused_kernel:
            assert (
                self.parallel_state.context_parallel_world_size > 1
            ), "cp_size should be greater than 1 when using fused kernel"

        if self.parallel_state.context_parallel_world_size > 1:
            if self.config.use_lasp:
                self._attention_impl = chunk_gla_global
            else:
                self._attention_impl = context_parallel_lightning_attention
        else:
            self._attention_impl = chunk_gla_global

        self.const_cache = inference_cache_mgr.get(CacheType.CONST_CACHE)
        self.is_sparsela: bool = config.is_sparsela
        self.sparse_rate: float = config.sparse_rate


    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        slope_rate: torch.Tensor,
        input_permuted: bool = True,
    ):

        if self.const_cache.enabled:
            origin_dtype = q.dtype
            res = self.do_generation(q, k, v, slope_rate, input_permuted)
            return res.to(origin_dtype)
        else:
            return self.do_forward(q, k, v, slope_rate, input_permuted)

    def do_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        slope_rate: torch.Tensor,
        input_permuted: bool = True,
    ) -> torch.Tensor:
        if not input_permuted and self.is_lasp():
            q, k, v = (rearrange(x, 's b h d -> b h s d') for x in (q, k, v))

        # [h, 1, 1] -> [B, H, T_device, K]
        gk = - slope_rate.expand_as(q).to(torch.float32)
        sparse_head_dim = int(self.sparse_rate * q.shape[-1])
        if self.is_sparsela:
            _, ind = torch.topk(torch.abs(k), sparse_head_dim, dim=-1, largest=False, sorted=False, out=None)
            k = k.scatter(-1, ind, torch.zeros_like(k))
            gk = gk.scatter(-1, ind, torch.zeros_like(gk))

        # gk = gk.to(q.dtype)
        org_type = q.dtype
        if self._attention_impl is context_parallel_lightning_attention:
            # [H, 1, 1] -> [B, H, T_device, K]
            gk = torch.chunk(
                gk,
                self.parallel_state.context_parallel_world_size,
                dim=1,  # Partition the head dimension.
            )[self.parallel_state.context_parallel_rank]
        q,k,v,gk = q.float(),k.float(),v.float(),gk.float()
        output = self._attention_impl(
            q,
            k,
            v,
            gk,
            self.parallel_state.context_parallel_group,
        ) # []
        output = output.to(org_type)

        if self._attention_impl is not context_parallel_lightning_attention:
            output = rearrange(output, 'b h s d -> s b (h d)')

        return output

    def do_generation(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        slope_rate: torch.Tensor,
        input_permuted: bool = True,
    ) -> torch.Tensor:
        if not input_permuted:
            q, k, v = (rearrange(x, 's b h d -> b h s d') for x in (q, k, v))

        # [h, 1, 1] -> [B, H, T_device, K]
        gk = - slope_rate.expand_as(q).to(torch.float32)
        sparse_head_dim = int(self.sparse_rate * q.shape[-1])
        if self.is_sparsela:
            _, ind = torch.topk(torch.abs(k), sparse_head_dim, dim=-1, largest=False, sorted=False, out=None)
            k = k.scatter(-1, ind, torch.zeros_like(k))
            gk = gk.scatter(-1, ind, torch.zeros_like(gk))

        q, k, v = (x.to(torch.float32) for x in (q, k, v))
        if self.const_cache.first_inference:
            batch_size = q.shape[0]
            if batch_size > 1:
                raise NotImplementedError(
                    "SparseLA generation prefill does not support padded "
                    "batches; batch size must be 1."
                )

            prompt_len = self.const_cache.prompt_len.item()
            ori_len = q.shape[2]
            q, k, v = (
                q[:, :, :prompt_len, :],
                k[:, :, :prompt_len, :],
                v[:, :, :prompt_len, :],
            )
            gk = gk.to(
                torch.bfloat16
            )  # prefill slope rate align with train accuracy
            out = self.prefill_single(q, k, v, gk)
            if ori_len == prompt_len:
                return out
            return F.pad(out, (0, 0, 0, 0, 0, ori_len - prompt_len))
        else:
            gk = gk.to(torch.float32)
            return self.decode_batch(q, k, v, gk)

    def prefill_single(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        slope_rate: torch.Tensor,
    ) -> torch.Tensor:
        _, h, _, d = q.shape
        e = d
        kv_history = self.const_cache.get_batch_inference_kv_memory()
        kv_history.zero_()
        if self.parallel_state.context_parallel_world_size > 1:
            raise NotImplementedError(
                "SparseLA generation prefill with context parallelism is "
                "not implemented."
            )
        output, new_kv = chunk_gla(q, k, v, slope_rate, scale=1, initial_state=None, output_final_state=True)
        # new_kv = new_kv[:, :, -1, :, :]
        self.const_cache.update_batch_inference_kv_memory(new_kv)
        new_kv = new_kv.reshape(h, d, e)
        output = rearrange(output, 'b h s d -> s b (h d)')
        return output

    def decode_batch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        slope_rate: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched decoding process for linear attention.
        q/k/v.shape = (batch_size, n_head, seq(1), head_dim)
        pre_kv.shape = (batch_size, n_head, head_dim, head_dim)
        """
        q_t = q[:, :, 0: 1, :]
        k_t = k[:, :, 0: 1, :]
        v_t = v[:, :, 0: 1, :]

        ratio = torch.exp(-slope_rate).transpose(-2, -1) # [B, H, 1, K] -> [B, H, K, 1]
        pre_kv = self.const_cache.get_batch_inference_kv_memory()
        cur_kv = torch.einsum(
            "... s d, ... s e -> ... d e", k_t, v_t, 
        ) # [B, H, K, V]
        new_kv = ratio * pre_kv + cur_kv # [B, H, K, 1] * [B, H, K, V]
        self.const_cache.update_batch_inference_kv_memory(new_kv)
        output = torch.einsum(
            "... s e, ... e d -> ... s d", q_t, new_kv
        )
        output = rearrange(output, "b h s d -> s b (h d)")
        return output

    def is_lasp(self):
        return self._attention_impl is chunk_gla_global


@dataclass
class DefaultLinearAttentionSpec(MeepoModuleSpec):
    config: DefaultLinearAttentionConfig
    module = DefaultLinearAttention
    submodule_spec = None
