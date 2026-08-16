# Figure 6: Long-context core integrations

This directory releases selected core components used to integrate SparseLA-H into inference and sequence-parallel training stacks.

## Scope

- [`inference/`](inference/) contains a SparseLA-H model implementation, SparseLA kernels, and a vLLM model-registry patch for a fixed vLLM revision.
- [`sequence_parallel/`](sequence_parallel/) contains a `meepo`-integrated SparseLA/linear-attention module and distributed Triton kernels.

This is **not** a complete end-to-end long-context training or evaluation system. The release does not include the 60B SparseLA-H configuration/checkpoint, tokenizer, pretraining data mixture, data upsampling pipeline, optimizer/scheduler, distributed launcher, variable-length ring-attention implementation, NIAH/RULER harness, or cluster orchestration used for the paper’s 4M-token training and 50M-token evaluation.

## Paper context

The paper describes SparseLA-H as 48 blocks with self-attention every eight blocks, 64 heads with head dimension 128, hidden dimension 3,072, and 32 experts per layer. The reported model has 60B total parameters and activates 9B per token. Those values describe the paper model; they are not defaults guaranteed by the released modules.

## Integration paths

Choose one path and follow its compatibility requirements:

- [vLLM inference integration](inference/README.md)
- [Sequence-parallel module integration](sequence_parallel/README.md)

Both paths assume a larger external codebase. Copying these files into an arbitrary recent framework version is not expected to work because internal APIs and neighboring modules are version-specific.

[Back to Figure 6](../README.md)
