from dataclasses import dataclass
import math
from typing import Optional, Tuple, Union

import torch

from meepo import argman
from meepo.dcpro.placeholder.dynamic_deduced import set_deduced_value
from meepo.dist_checkpointing.mapping import (
    ShardedStateDict,
    ShardedTensor,
    ShardedTensorFactory,
)
from meepo.module._base_module import MeepoModule, MeepoModuleBaseConfig
from meepo.module.linear import (
    ColumnParallelLinear,
    ColumnParallelLinearSpec,
    ParallelLinearSpec,
    RowParallelLinearSpec,
)
from meepo.module.norm import RMSNormSpec, SRMSNormSpec
from meepo.module.utils import (
    MeepoModuleSpec,
    MeepoSubmoduleSpec,
    build_fn_for_split,
    build_module,
    merge_fn_for_split,
)
from meepo.parallelism import checkpoint
from meepo.utils import numerical
from meepo.utils.inference_cache_mgr import CacheType, inference_cache_mgr
from meepo.utils.meepo_recorder import recorder

from .impl import (
    DefaultLinearAttentionSpec,
    fused_qkv_gate_transpose,
    naive_qkv_preprocess,
)


@dataclass
class LinearAttentionConfig(MeepoModuleBaseConfig):
    # LinearAttentionConfig
    num_attention_heads: int = argman.num_attention_heads
    kv_channels: int = argman.kv_channels
    group_query_attention: bool = argman.group_query_attention
    num_query_groups: int = argman.num_query_groups
    slope_rate_scaler: float = 0.0
    enable_recompute: bool = False

    def validate(self):
        super().validate()

        assert (
            self.group_query_attention is False
        ), "Group query attention is not supported for LinearAttention yet"

        if not self.group_query_attention:
            assert (
                self.num_query_groups == self.num_attention_heads
                or self.num_query_groups == -1
            )


@dataclass
class LinearAttentionSubmoduleSpec(MeepoSubmoduleSpec):
    linear_qkv_gate: Union[ColumnParallelLinearSpec, ParallelLinearSpec]
    attention: DefaultLinearAttentionSpec
    attn_norm: Union[SRMSNormSpec, RMSNormSpec]
    linear_out: Union[RowParallelLinearSpec, ParallelLinearSpec]


class LinearAttention(MeepoModule):
    def __init__(
        self,
        config: LinearAttentionConfig,
        submodule_spec: LinearAttentionSubmoduleSpec,
    ):
        super().__init__(config, submodule_spec)
        assert isinstance(config, LinearAttentionConfig)
        assert isinstance(submodule_spec, LinearAttentionSubmoduleSpec)
        self.config: LinearAttentionConfig
        self.submodule_spec: LinearAttentionSubmoduleSpec

        # Here, kv_channels means the head_dim, num_attention_heads means head_num, num_query_groups means group_num
        self.head_dim = self.config.kv_channels
        self.num_heads_per_partition = numerical.ensure_divisibility_and_divide(
            self.config.num_attention_heads,
            self.parallel_state.tensor_model_parallel_world_size,
        )
        if self.config.group_query_attention:
            self.num_groups_per_partition = (
                numerical.ensure_divisibility_and_divide(
                    self.config.num_query_groups,
                    self.parallel_state.tensor_model_parallel_world_size,
                )
            )
        else:
            self.num_groups_per_partition = self.num_heads_per_partition

        self.linear_qkv_gate = build_module(
            self.submodule_spec.linear_qkv_gate
        )
        self.attention = build_module(self.submodule_spec.attention)
        self.attn_norm = build_module(self.submodule_spec.attn_norm)
        self.linear_out = build_module(self.submodule_spec.linear_out)

        self.qkv_gate_split_size = [
            (self.kv_projection_size * 2 + self.q_projection_size)
            // self.parallel_state.tensor_model_parallel_world_size,
            self.q_projection_size
            // self.parallel_state.tensor_model_parallel_world_size,
        ]

        slope_rate_heads = self.config.num_attention_heads
        slope_rate = self._build_slope_tensor(slope_rate_heads)
        slope_rate = (
            slope_rate.to(torch.cuda.current_device())
            * self.config.slope_rate_scaler
        )
        if self.parallel_state.tensor_model_parallel_world_size > 1:
            self.slope_rate = torch.chunk(
                slope_rate,
                self.parallel_state.tensor_model_parallel_world_size,
                0,
            )[self.parallel_state.tensor_model_parallel_rank]
        else:
            self.slope_rate = slope_rate

        self.const_cache = inference_cache_mgr.get(CacheType.CONST_CACHE)

        # fused ops only support num_query_groups == num_attention_heads
        self._use_naive_qkv_process = (
            self.num_groups_per_partition != self.num_heads_per_partition
            or not self.attention.is_lasp()
        )

    @staticmethod
    def _build_slope_tensor(n_attention_heads: int):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(
                    n
                )  # In the paper, we only train models that have 2^a heads for some a. This function has
            else:  # some good properties that only occur when the input is a power of 2. To maintain that even
                closest_power_of_2 = 2 ** math.floor(
                    math.log2(n)
                )  # when the number of heads is not a power of 2, we use this workaround.
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][
                        : n - closest_power_of_2
                    ]
                )

        # h, 1, 1
        slopes = torch.tensor(get_slopes(n_attention_heads)).reshape(
            n_attention_heads, 1, 1
        )

        return slopes

    def forward(
        self,
        input_tensor: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        cu_seq_len: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        is_first_microbatch: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Self-Attention Module

        Args:
            input_tensor(torch.Tensor): input tensor
            attention_mask(torch.Tensor): attention mask
            rotary_pos_emb(Optional[torch.Tensor]): rotary position embedding
            cu_seq_len(Optional[int]): cumulative sequence length
            max_seqlen(Optional[int]): maximum sequence length
            is_first_microbatch(Optional[bool]): whether it is the first microbatch
        Returns:
            output(torch.Tensor): output tensor
        Shape:
            - input_tensor: [seq_len // tp_size, batch, hidden] if sequence_parallel else [seq_len, batch, hidden]
            - attention_mask: [batch, seq_len, seq_len]
            - output: [seq_len // tp_size, batch, hidden] if sequence_parallel else [seq_len, batch, hidden]
        """
        if self.const_cache.enabled:
            self.const_cache.set_attention_config(
                self.num_groups_per_partition, self.head_dim, input_tensor.dtype
            )

        def custom_forward(qkv_gate_layer, is_forward=True):
            if self._use_naive_qkv_process:
                qkv_layer, gate_layer = torch.split(
                    qkv_gate_layer, self.qkv_gate_split_size, dim=-1
                )
                q_layer, k_layer, v_layer, gate_layer = naive_qkv_preprocess(
                    qkv_layer,
                    gate_layer,
                    self.num_heads_per_partition,
                    self.num_groups_per_partition,
                    self.head_dim,
                )
            else:
                (
                    q_layer,
                    k_layer,
                    v_layer,
                    gate_layer,
                ) = fused_qkv_gate_transpose(
                    qkv_gate_layer,
                    self.num_heads_per_partition,
                    self.num_groups_per_partition,
                )

            # [seqlen, batch, num_heads_per_partition * head_dim]
            out_layer = self.attention(
                q=q_layer,
                k=k_layer,
                v=v_layer,
                slope_rate=self.slope_rate,
                input_permuted=not self._use_naive_qkv_process,
            )
            out_layer = self.attn_norm(out_layer)
            out_layer = gate_layer * out_layer
            if is_forward:
                recorder.report_tensors(
                    ["q_layer", "k_layer", "v_layer", "gate_out", "gate_layer"],
                    [
                        q_layer,
                        k_layer,
                        v_layer,
                        gate_layer.view(
                            gate_layer.shape[0],
                            gate_layer.shape[1],
                            -1,
                            self.head_dim,
                        ),
                        out_layer.view(
                            gate_layer.shape[0],
                            gate_layer.shape[1],
                            -1,
                            self.head_dim,
                        ),
                    ],
                    name_prefix="activation/",
                    add_layer_info=True,
                )
            return out_layer

        # [seqlen, batch, hidden] --> [seqlen, batch, hidden_per_partition]
        # Here hidden_per_partition equals
        #   'num_groups_per_partition * (num_heads_per_partition // num_groups_per_partition + 2) * head_dim)'
        # Contains query's hidden as 'num_heads_per_partition * head_dim'
        # Contains key's   hidden as 'num_groups_per_partition * head_dim'
        # Contains value's hidden as 'num_groups_per_partition * head_dim'

        qkv_gate_layer, _ = self.linear_qkv_gate(input_tensor)
        # qkv_layer, gate_layer = torch.split(
        #     qkv_gate_layer, self.qkv_gate_split_size, dim=-1
        # )

        if self.config.enable_recompute:
            out_layer = checkpoint(
                custom_forward,
                qkv_gate_layer,
                is_forward=True,
                use_reentrant=True,
            )
        else:
            out_layer = custom_forward(qkv_gate_layer)

        output, bias = self.linear_out(out_layer)

        recorder.report_tensors(
            ["attn_linear_out"],
            [output],
            name_prefix="activation/",
            add_layer_info=True,
        )
        return output, bias

    def _update_submodule_spec_from_config(self):
        # Here, kv_channels means the head_dim, num_attention_heads means head_num, num_query_groups means group_num
        self.q_projection_size = (
            self.config.kv_channels * self.config.num_attention_heads
        )
        if self.config.group_query_attention:
            self.kv_projection_size = (
                self.config.kv_channels * self.config.num_query_groups
            )
        else:
            self.kv_projection_size = self.q_projection_size

        linear_qkv_gate_config = self.submodule_spec.linear_qkv_gate.config
        # assert linear_qkv_gate_config.is_moe_linear is False
        assert linear_qkv_gate_config.skip_bias_add is False
        # assert linear_qkv_gate_config.gather_output is False
        set_deduced_value(
            linear_qkv_gate_config,
            "output_size",
            2 * self.kv_projection_size
            + self.q_projection_size
            + self.q_projection_size,
        )

        # linear_gate_config = self.submodule_spec.linear_gate.config
        # assert linear_qkv_gate_config.is_moe_linear is False
        # assert linear_qkv_gate_config.skip_bias_add is False
        # assert linear_qkv_gate_config.gather_output is False
        # set_deduced_value(
        #     linear_gate_config,
        #     "output_size",
        #     self.q_projection_size,
        # )

        linear_out_config = self.submodule_spec.linear_out.config
        # assert linear_out_config.is_moe_linear is False
        assert linear_out_config.skip_bias_add is False
        # assert linear_out_config.input_is_parallel is True
        set_deduced_value(
            linear_out_config, "input_size", self.q_projection_size
        )

        attn_norm_config = self.submodule_spec.attn_norm.config
        # SRMSNorm and RMSNorm expose different configuration fields; only
        # update parallel_norm when the selected implementation supports it.
        if hasattr(attn_norm_config, "parallel_norm"):
            if self.parallel_state.tensor_model_parallel_world_size > 1:
                set_deduced_value(attn_norm_config, "parallel_norm", True)
            else:
                set_deduced_value(attn_norm_config, "parallel_norm", False)

        self.submodule_spec.linear_qkv_gate.config = linear_qkv_gate_config
        self.submodule_spec.linear_out.config = linear_out_config
        self.submodule_spec.attn_norm.config = attn_norm_config

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        sharded_state_dict = {}
        for name, module in self._modules.items():
            if name == 'linear_qkv_gate':
                sub_sd = self._sharded_state_dict_for_qkv_gate(
                    name, module, prefix, sharded_offsets, metadata
                )
            else:
                sub_sd = module.sharded_state_dict(
                    f'{prefix}{name}.', sharded_offsets, metadata
                )
            sharded_state_dict.update(sub_sd)
        return sharded_state_dict

    def _sharded_state_dict_for_qkv_gate(
        self,
        module_name: str,
        module: torch.nn.Module,
        prefix: str,
        sharded_offsets: Tuple[Tuple[int, int, int]],
        metadata: Optional[dict] = None,
    ):
        assert module_name == 'linear_qkv_gate', module_name
        sharded_state_dict = module.sharded_state_dict(
            f'{prefix}{module_name}.', sharded_offsets, metadata
        )
        if isinstance(module, ColumnParallelLinear):
            name_prefix = f'{prefix}{module_name}.column_linear'
        else:
            name_prefix = f'{prefix}{module_name}'
        weight_keys = [f"{name_prefix}.weight"]
        if hasattr(module, "weight_scale") and hasattr(
            module, "weight_zero_point"
        ):
            weight_keys = weight_keys + [
                f"{name_prefix}.weight_scale",
                f"{name_prefix}.weight_zero_point",
            ]

        tp_rank = self.parallel_state.tensor_model_parallel_rank
        tp_size = self.parallel_state.tensor_model_parallel_world_size

        build_kwargs = {
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "split_size": self.qkv_gate_split_size,
            "sharded_offsets": sharded_offsets,
        }
        merge_kwargs = {}

        for weight_key in weight_keys:
            prev_sh_ten: ShardedTensor = sharded_state_dict[weight_key]

            # We must split the tensor into 2 parts, each sharded separately.
            # This requires a ShardedTensorFactory which `chunk`s during saving
            # and `cat`s during loading
            sharded_state_dict[weight_key] = ShardedTensorFactory(
                prev_sh_ten.key,
                prev_sh_ten.data,
                build_fn_for_split,
                merge_fn_for_split,
                prev_sh_ten.replica_id,
                prev_sh_ten.comm,
                build_kwargs,
                merge_kwargs,
            )
        return sharded_state_dict


@dataclass
class LinearAttentionSpec(MeepoModuleSpec):
    config: LinearAttentionConfig
    submodule_spec: LinearAttentionSubmoduleSpec
    module = LinearAttention
