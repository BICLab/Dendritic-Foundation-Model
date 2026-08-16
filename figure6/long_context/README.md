# Figure 6: SparseLA-H long-context training and inference

This directory contains SparseLA-H modules for vLLM inference and
sequence-parallel training.

## Components

- [`inference/`](inference/) contains a SparseLA-H model implementation, SparseLA kernels, and a vLLM model-registry patch for a fixed vLLM revision.
- [`sequence_parallel/`](sequence_parallel/) contains a `meepo`-integrated SparseLA/linear-attention module and distributed Triton kernels.

## SparseLA-H architecture

In our paper, SparseLA-H consists of 48 blocks, with self-attention inserted
every eight blocks. The model uses 64 heads with head dimension 128, a hidden
dimension of 3,072, and 32 experts per layer, for a total of 60B parameters
with 9B activated per token.

## Integration paths

Choose one path and follow its compatibility requirements:

- [vLLM inference integration](inference/README.md)
- [Sequence-parallel module integration](sequence_parallel/README.md)

Both paths integrate with a host framework. Use the framework versions
documented in each subdirectory because the relevant APIs are version-specific.

[Back to Figure 6](../README.md)
