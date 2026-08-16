# Figure 6: Language modeling, efficiency, and long context

Figure 6 evaluates SparseLA and SparseLA-H on language modeling, efficiency, and context extension.

## Overview

The paper reports:

- 400M-, 1B-, and 3B-parameter language models pretrained at an 8k-token context length and evaluated on Commonsense Reasoning, SCROLLS, and Needle-in-a-Haystack (NIAH);
- operator-level training efficiency on NVIDIA A100 GPUs and network-level inference comparisons;
- the 60B-parameter SparseLA-H architecture with 48 blocks, self-attention inserted every eight blocks, 64 heads of dimension 128, hidden dimension 3,072, and 32 experts per layer, with 9B parameters activated per token;
- context-window extension during training to 4M tokens and NIAH evaluation up to 50M tokens.

The code is organized into foundation-model architecture implementations and
SparseLA-H modules for long-context training and inference.

## Code

| Component | Contents | Paper experiment |
| --- | --- | --- |
| [Model comparison](model_comparison/) | Implementations of LLaMA2, TransNormer3, HGRN2, Mamba, Mamba2, MetaLA, and SparseLA | Language understanding, reasoning, retrieval, and efficiency comparisons |
| [Long-context SparseLA-H](long_context/) | vLLM inference and sequence-parallel training modules | Context extension and long-range retrieval |

Architecture width, depth, state dimensions, and feed-forward dimensions are
configured through each model’s JSON configuration.

## Environment

Use separate environments:

- model definitions: PyTorch, Transformers, architecture-specific CUDA/Triton dependencies;
- inference integration: the exact vLLM commit documented in the [inference README](long_context/inference/README.md);
- sequence parallelism: the internal `meepo` framework plus PyTorch distributed, CUDA, Triton, and einops.

## Start here

- For foundation-model implementations and configuration examples, see
  [model comparison](model_comparison/README.md).
- For SparseLA-H training and inference modules, see
  [long-context integrations](long_context/README.md).

[Back to repository overview](../README.md)
