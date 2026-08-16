from dataclasses import dataclass

import torch

from meepo.utils.metaclass import SingletonMeta


class ConstCache(metaclass=SingletonMeta):
    def __init__(self):
        self.enabled = False

    def init(self, max_batch_size, max_sequence_len):
        self.max_batch_size = max_batch_size
        # layer_num -> kv of each head
        self.kv_memory_dict = {}
        self.first_inference = True
        self.enabled = False

        @dataclass
        class LayerContext:
            layer_num: int
            param_type: torch.dtype
            num_query_groups_per_partition: int
            hidden_size_per_attention_head: int

        # info of current running layer
        self.ctx = LayerContext(0, None, 0, 0)

    def set_batch_idx(self, batch_idx):
        self.cache_batch_idx = batch_idx

    def set_cache_seqlens(self, cache_seqlens):
        self.cache_seqlens = cache_seqlens
        zero_tensor = torch.zeros_like(cache_seqlens)
        self.first_inference = torch.equal(zero_tensor, cache_seqlens)

    def set_prompt_len(self, prompt_len: int):
        self.prompt_len = prompt_len

    def set_attention_config(
        self,
        num_query_groups_per_partition: int,
        hidden_size_per_attention_head: int,
        param_type: torch.dtype,
    ):
        self.ctx.num_query_groups_per_partition = num_query_groups_per_partition
        self.ctx.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.ctx.param_type = param_type

    def set_layer(self, layer_num: int):
        self.ctx.layer_num = layer_num
        self.ctx.param_type = None
        self.ctx.num_query_groups_per_partition = 0
        self.ctx.hidden_size_per_attention_head = 0

    def get_layer_inference_kv_memory(self):
        layer_num = self.ctx.layer_num
        if layer_num not in self.kv_memory_dict:
            self.kv_memory_dict[layer_num] = self._allocate_memory()

        return self.kv_memory_dict[layer_num]

    def get_batch_inference_kv_memory(self):
        layer_kv_memory = self.get_layer_inference_kv_memory()
        return layer_kv_memory[self.cache_batch_idx]

    def update_batch_inference_kv_memory(self, result):
        layer_kv_memory = self.get_layer_inference_kv_memory()
        layer_kv_memory[self.cache_batch_idx] = result

    def _allocate_memory(self):
        """
        Allocate memory to store linear attention const cache during inference.
        shape: [b, n_head, d, d]
        """
        return torch.empty(
            self.max_batch_size,
            self.ctx.num_query_groups_per_partition,
            self.ctx.hidden_size_per_attention_head,
            self.ctx.hidden_size_per_attention_head,
            dtype=torch.float32,
            device=torch.cuda.current_device(),
        )

    def replicate_cache(self, exist_batch_size: int, repeat_times: int):
        total_batch_size = exist_batch_size * repeat_times
        assert total_batch_size <= self.max_batch_size

        for memory in self.kv_memory_dict.values():
            for i in range(1, repeat_times):
                start_idx = i * exist_batch_size
                end_idx = (i + 1) * exist_batch_size
                memory[start_idx:end_idx] = memory[:exist_batch_size]

    def disable(self):
        self.kv_memory_dict = {}
        self.enabled = False

    def enable(self):
        self.enabled = True


const_cache = ConstCache()
