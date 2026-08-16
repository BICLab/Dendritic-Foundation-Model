# Figure 6: Model architecture comparison

This directory contains size-independent architecture implementations and
example configurations used for the Figure 6 language-model comparison. The
architecture code can be configured at different model scales; the bundled
JSON files record the approximately 400M settings used as concrete examples.
No model weights are included.

## Contents

| Directory | Architecture class | Example configuration |
| --- | --- | --- |
| [`sparsela/`](sparsela/) | `SparseLAForCausalLM` | [`config-example-378m-300B.json`](sparsela/config-example-378m-300B.json) |
| [`metala/`](metala/) | `MetaLAForCausalLM` | [`config-example-378m-300B.json`](metala/config-example-378m-300B.json) |
| [`mamba/`](mamba/) | `MambaForCausalLM` | [`config-example-378m-300B.json`](mamba/config-example-378m-300B.json) |
| [`mamba2/`](mamba2/) | `Mamba2ForCausalLM` | [`config-example-378m-300B.json`](mamba2/config-example-378m-300B.json) |
| [`hgrn2/`](hgrn2/) | `Hgrn2ForCausalLM` | [`config-example-385m-300B.json`](hgrn2/config-example-385m-300B.json) |
| [`tnl3/`](tnl3/) | `TransnormerForCausalLM` | [`config-example-410m-300B.json`](tnl3/config-example-410m-300B.json) |
| [`llama2/`](llama2/) | `LlamaForCausalLM` | [`config-example-410m-300B.json`](llama2/config-example-410m-300B.json) |

Each directory contains one architecture implementation,
`generation_config.json`, and the corresponding example model configuration;
most also include a custom configuration class and helper modules. The
`378m`–`410m` labels describe example parameter counts, while `300B` records
the associated pretraining-token setting. Neither label limits the model
implementation.

## Environment

Use a dedicated CUDA environment with PyTorch and Hugging Face Transformers. The checked-in configurations record Transformers versions around 4.38–4.44, depending on the architecture; treat those values as compatibility clues, not as a tested unified lockfile.

Architecture-specific imports may require:

- einops;
- Triton and Flash Linear Attention (`fla`);
- `causal-conv1d` and `mamba-ssm`;
- CUDA-capable hardware and matching compiled extensions.

Install only the dependencies required by the model being inspected. Custom modeling files may execute CUDA-specific imports even when the intended operation is only initialization.

## Random initialization

To instantiate the example SparseLA configuration with fresh random
parameters, run from the repository root:

```python
from figure6.model_comparison.sparsela.configuration_sparsela import SparseLAConfig
from figure6.model_comparison.sparsela.modeling_sparsela import SparseLAForCausalLM

config = SparseLAConfig.from_json_file(
    "figure6/model_comparison/sparsela/config-example-378m-300B.json"
)
model = SparseLAForCausalLM(config)
print(f"{sum(p.numel() for p in model.parameters()):,} parameters")
```

This executes the standard initialization path and does not load pretrained
weights. The other directories follow the same pattern using the
configuration/model class named by their `auto_map` and `architectures`
fields. Mamba and Mamba2 use the corresponding Transformers configuration
classes directly.

## Scaling to 1B or 3B

The paper also reports 1B- and 3B-parameter experiments. To configure those
scales:

1. copy the relevant example JSON and adjust architecture-specific width,
   depth, attention/state dimensions, and feed-forward dimensions;
2. instantiate the model and count parameters rather than relying on a filename;
3. preserve divisibility constraints for heads, tensor parallelism, and fused kernels;
4. retune batch size, learning rate, precision, activation checkpointing, and distributed parallelism;
5. train from random initialization on the intended corpus.

The repository does not provide the bilingual pretraining corpus, tokenizer assets, optimizer/scheduler state, distributed trainer, checkpoints, or downstream evaluation harness. Therefore these files alone cannot reproduce the reported pretrained-model scores.

## Configuration notes

The example configurations use an 8,192-token target/sample length and a vocabulary size of 100,280. Token IDs differ for HGRN2. Verify tokenizer compatibility before any language-model training or evaluation; no tokenizer is bundled.

[Back to Figure 6](../README.md)
