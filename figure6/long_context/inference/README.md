# SparseLA-H vLLM inference integration

This directory contains model, cache, kernel, and registry files for
integrating SparseLA-H with vLLM v0.9.1.

## Compatibility target

Use [vLLM v0.9.1](https://github.com/vllm-project/vllm/tree/v0.9.1). The cache
adapter relies on the constant-size cache API in that release.

The integration depends on PyTorch, CUDA, Triton, einops, Transformers, Flash Linear Attention (`fla`), and the dependencies required by that vLLM revision. Build vLLM using its source-install instructions for your CUDA/PyTorch platform.

## Components

- [`layers/sparse_la.py`](layers/sparse_la.py): SparseLA prefill/decode kernels.
- [`models/sparse_la_h.py`](models/sparse_la_h.py): SparseLA-H model definition for vLLM inference.
- [`models/sparse_la_h_cache.py`](models/sparse_la_h_cache.py): constant-size recurrent-state cache.
- [`patches/vllm_registry.patch`](patches/vllm_registry.patch): minimal model-registry addition.

## Integration

Clone and pin the exact vLLM revision:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.9.1
```

From the root of this repository, set the destination and copy the files:

```bash
export REPO_ROOT="$(pwd)"
export VLLM_SRC=/absolute/path/to/vllm

cp "$REPO_ROOT/figure6/long_context/inference/layers/sparse_la.py" \
  "$VLLM_SRC/vllm/model_executor/layers/sparse_la.py"
cp "$REPO_ROOT/figure6/long_context/inference/models/sparse_la_h.py" \
  "$VLLM_SRC/vllm/model_executor/models/sparse_la_h.py"
cp "$REPO_ROOT/figure6/long_context/inference/models/sparse_la_h_cache.py" \
  "$VLLM_SRC/vllm/model_executor/models/sparse_la_h_cache.py"

git -C "$VLLM_SRC" apply \
  "$REPO_ROOT/figure6/long_context/inference/patches/vllm_registry.patch"
```

Review the registry patch before applying it, especially if the vLLM checkout contains local changes. Then build/install vLLM from `VLLM_SRC` according to the tagged release’s documentation.

After installation, verify that vLLM recognizes the SparseLA-H model:

```bash
python -c "from vllm.model_executor.models.registry import ModelRegistry; print('SparseLAHForCausalLM' in ModelRegistry.get_supported_archs())"
```

[Back to long-context integrations](../README.md)
