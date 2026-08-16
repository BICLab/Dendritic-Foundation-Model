# SparseLA sequence-parallel engine

This directory contains the SparseLA modules and distributed kernels used by
the sequence-parallel engine.

## Components

- [`linear_attention.py`](linear_attention.py): [Meepo](https://github.com/MiniMax-AI) integration, QKV/gate projection, normalization, and output projection.
- [`const_cache.py`](const_cache.py): constant-cache support used by the integration.
- [`impl/default_attention/default_attention.py`](impl/default_attention/default_attention.py): SparseLA/linear-attention dispatch between local and context-parallel implementations.
- [`impl/qkv_gate_transpose.py`](impl/qkv_gate_transpose.py): QKV/gate preprocessing.
- [`impl/default_attention/`](impl/default_attention/): Triton kernels for chunked GLA, sequence/context parallelism, and inference helpers.

## Dependencies

The engine uses CUDA-enabled PyTorch, Triton, einops, `packaging`, Meepo APIs,
and initialized tensor/context-parallel process groups.

## Runtime options

- `SP_RATE` controls the retained SparseLA dimensions and defaults to `0.5`.
- `CHUNK_SIZE64` controls the global chunked-GLA chunk size and defaults to `64`.
- `FLA_USE_CUDA_GRAPH=1` enables CUDA-graph execution in supported helpers.

[Back to long-context integrations](../README.md)
