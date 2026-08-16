# Figure 6: Model architecture comparison

This directory contains implementations of the foundation-model architectures
compared in Figure 6. Each architecture is accompanied by a JSON configuration
corresponding to one of the paper’s 400M-scale experiments.

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
`378m`–`410m` labels record the parameter counts of the configurations, and
`300B` records the associated pretraining-token setting. Architecture
dimensions are controlled by the JSON fields.

## Environment

Use a dedicated CUDA environment with PyTorch and Hugging Face Transformers.
The example configurations record Transformers versions from 4.38 to 4.44,
reflecting the environments used for the individual architectures.

Architecture-specific imports may require:

- einops;
- Triton and Flash Linear Attention (`fla`);
- `causal-conv1d` and `mamba-ssm`;
- CUDA-capable hardware and matching compiled extensions.

Install the architecture-specific dependencies required by the model being
used.

## Model initialization

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

This executes the standard initialization path. To configure another model
scale, copy the example JSON and adjust the architecture-specific width,
depth, attention/state dimensions, and feed-forward dimensions, then
instantiate the model and verify its parameter count.

The other directories follow the same pattern using the configuration and
model classes named by their `auto_map` and `architectures` fields. Mamba and
Mamba2 use the corresponding Transformers configuration classes directly.

[Back to Figure 6](../README.md)
