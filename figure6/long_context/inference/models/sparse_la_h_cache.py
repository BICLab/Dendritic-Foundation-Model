# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Constant-size state cache for the SparseLA-H vLLM integration.

Adapted from vLLM v0.9.1's constant-size MiniMax cache implementation.
"""

from dataclasses import dataclass, field

import torch

from vllm.model_executor.models.constant_size_cache import ConstantSizeCache


@dataclass
class SparseLAHCacheParams:
    sparse_la_h_cache: torch.Tensor = field(default_factory=torch.Tensor)
    state_indices_tensor: torch.Tensor = field(default_factory=torch.Tensor)

    def at_layer_idx(self, layer_idx):
        return SparseLAHCacheParams(
            self.sparse_la_h_cache[layer_idx, ...],
            self.state_indices_tensor,
        )


class SparseLAHCacheManager(ConstantSizeCache):
    def __init__(self, dtype, cache_shape):
        super().__init__(cache_shape[1])
        self._sparse_la_h_cache = torch.empty(
            size=cache_shape,
            dtype=dtype,
            device="cuda",
        )

    @property
    def cache(self):
        return self._sparse_la_h_cache

    def _copy_cache(self, from_index: int, to_index: int):
        assert len(self.cache) > 0
        for layer_cache in self.cache:
            layer_cache[to_index].copy_(
                layer_cache[from_index],
                non_blocking=True,
            )
