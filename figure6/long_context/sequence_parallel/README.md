# SparseLA sequence-parallel core module

This directory contains selected SparseLA/linear-attention modules and distributed kernels used inside a larger sequence-parallel training stack. It is not a standalone trainer.

## Components

- [`linear_attention.py`](linear_attention.py): `meepo` module/configuration integration, QKV/gate projection, normalization, and output projection.
- [`const_cache.py`](const_cache.py): constant-cache support used by the integration.
- [`impl/default_attention/default_attention.py`](impl/default_attention/default_attention.py): SparseLA/linear-attention dispatch between local and context-parallel implementations.
- [`impl/qkv_gate_transpose.py`](impl/qkv_gate_transpose.py): QKV/gate preprocessing.
- [`impl/default_attention/`](impl/default_attention/): Triton kernels for chunked GLA, sequence/context parallelism, and inference helpers.

## Dependencies

The code imports the external `meepo` framework throughout. That framework, its launcher, model stack, distributed process-group setup, and configuration system are not included in this repository. Integration also requires:

- a compatible CUDA-enabled PyTorch build;
- Triton, einops, and `packaging`;
- the `meepo` APIs referenced by the imports;
- initialized tensor/context-parallel process groups and compatible collective communication support.

Because this module relies on internal framework APIs, integration should use
the dependency versions selected by the host `meepo` training stack.

## Required configuration

The module provides conservative defaults, which can be overridden before
process startup:

```bash
export SP_RATE=0.5
export CHUNK_SIZE64=64
export FLA_USE_CUDA_GRAPH=0
```

- `SP_RATE` defaults to `0.5` and controls the retained SparseLA dimensions. Choose the value used by the target experiment rather than assuming the default reproduces the paper.
- `CHUNK_SIZE64` defaults to `64` in the global chunked-GLA path.
- `FLA_USE_CUDA_GRAPH=1` enables CUDA-graph behavior in a helper when supported; the default behavior is disabled.

Distributed rank/world-size variables and device placement are managed by the host `meepo`/PyTorch distributed launcher, not by a script in this directory.

## Integration outline

1. Start from the compatible parent `meepo` training codebase.
2. Place this package at the module path expected by imports such as `meepo.module.linear_attention`.
3. Register `LinearAttentionConfig`, `LinearAttentionSubmoduleSpec`, and `DefaultLinearAttentionSpec` in the parent model specification.
4. Configure tensor and context parallel process groups.
5. Configure `SP_RATE` and `CHUNK_SIZE64` before process startup when the defaults are not appropriate.
6. Validate numerical agreement against a non-partitioned reference on small tensors before scaling.

There is no executable training entry point, model configuration, data loader, optimizer, checkpointing workflow, or end-to-end test in this directory.

## Release boundary

The paper’s 4M-token training and 50M-token NIAH result also depend on SparseLA-H model assembly, periodic self-attention, variable-length ring attention, data-mixture upsampling, distributed optimization, and evaluation infrastructure. Those components are not released here. This directory should therefore be cited as a core sequence-parallel integration module, not as the complete long-context training system.

[Back to long-context integrations](../README.md)
