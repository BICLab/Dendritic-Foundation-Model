# Figure 6: Language modeling, efficiency, and long context

Figure 6 evaluates SparseLA and SparseLA-H on language modeling, efficiency, and context extension.

## Paper scope

The paper reports:

- 400M-, 1B-, and 3B-parameter language models pretrained at an 8k-token context length and evaluated on Commonsense Reasoning, SCROLLS, and Needle-in-a-Haystack (NIAH);
- operator-level training efficiency on NVIDIA A100 GPUs and network-level inference comparisons;
- the 60B-parameter SparseLA-H architecture with 48 blocks, self-attention inserted every eight blocks, 64 heads of dimension 128, hidden dimension 3,072, and 32 experts per layer, with 9B parameters activated per token;
- context-window extension during training to 4M tokens and NIAH evaluation up to 50M tokens.

The released code focuses on architecture definitions and integration
components; pretrained artifacts and cluster infrastructure are outside the
release scope.

## Released code

| Component | Contents | Important boundary |
| --- | --- | --- |
| [Model comparison](model_comparison/) | Seven size-independent architecture implementations with approximately 378M–410M example configurations | Configurations and source only; no pretrained weights or full pretraining pipeline |
| [Long-context integration](long_context/) | vLLM inference files and a sequence-parallel SparseLA module | Core integration modules only; not a complete end-to-end long-context system |

The approximately 400M examples provide starting configurations. Configuring
the 1B and 3B variants requires scaling architecture fields, recalculating
parameter counts, and retuning optimization and parallelism before training
from random initialization on the intended corpus.

## Environment

Use separate environments:

- model definitions: PyTorch, Transformers, architecture-specific CUDA/Triton dependencies;
- inference integration: the exact vLLM commit documented in the [inference README](long_context/inference/README.md);
- sequence parallelism: the internal `meepo` framework plus PyTorch distributed, CUDA, Triton, and einops.

No model weights, tokenizer files, pretraining corpus, distributed launch stack, or benchmark harness is included at this level.

## Start here

- To inspect or randomly initialize an approximately 400M model, see [model comparison](model_comparison/README.md).
- To understand the long-context integration scope, see [long-context integrations](long_context/README.md).

[Back to repository overview](../README.md)
