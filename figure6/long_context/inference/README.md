# SparseLA-H vLLM inference integration

This directory contains the model and kernel files used to add SparseLA-H support to a specific vLLM source revision. It is an integration patch, not a standalone inference package.

## Compatibility target

Use [vLLM v0.9.1](https://github.com/vllm-project/vllm/tree/v0.9.1). The released cache adapter relies on the constant-size cache API introduced in that release. vLLM internal APIs change frequently; compatibility with another revision is not implied.

The integration depends on PyTorch, CUDA, Triton, einops, Transformers, Flash Linear Attention (`fla`), and the dependencies required by that vLLM revision. Build vLLM using its source-install instructions for your CUDA/PyTorch platform.

## Released files

- [`layers/sparse_la.py`](layers/sparse_la.py): SparseLA prefill/decode kernels.
- [`models/sparse_la_h.py`](models/sparse_la_h.py): inference-only SparseLA-H model definition.
- [`models/sparse_la_h_cache.py`](models/sparse_la_h_cache.py): constant-size recurrent-state cache.
- [`patches/vllm_registry.patch`](patches/vllm_registry.patch): minimal model-registry addition.

No checkpoint, model `config.json`, tokenizer, serving example, or evaluation harness is included.

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

## Model requirements

To load a model, an external Hugging Face-style model directory must provide:

- a configuration whose `architectures` includes `SparseLAHForCausalLM`;
- tokenizer files;
- weights with names and tensor shapes expected by `models/sparse_la_h.py`;
- model dimensions and parallelism settings compatible with the selected hardware.

This repository does not provide those artifacts. Random initialization is not a substitute for the paper’s trained SparseLA-H checkpoint, and the approximately 400M examples under [`../../model_comparison/`](../../model_comparison/) are different model-comparison configurations.

## Validation

After installation, first verify that the patched modules import in the vLLM environment:

```bash
python -c "from vllm.model_executor.models.registry import ModelRegistry; print('SparseLAHForCausalLM' in ModelRegistry.get_supported_archs())"
```

Successful registration confirms only that the integration is discoverable. End-to-end generation additionally requires compatible external weights/configuration and GPU kernel support.

Common runtime controls such as `CUDA_VISIBLE_DEVICES` are supplied by the host environment. This patch does not define custom dataset or checkpoint path environment variables; pass model paths through the normal vLLM interface.

[Back to long-context integrations](../README.md)
