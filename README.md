# Dendritic computation bridges neuroscience and foundation models

Official code release for the paper **“Dendritic computation bridges neuroscience and foundation models.”**

Repository: <https://github.com/BICLab/Dendritic-Foundation-Model>

## Overview

This work connects simplified dendritic computation with attention mechanisms in foundation models. It uses dendritic organization as a structural lens for linear attention, introduces the sparse-memory operator **SparseLA**, and studies the hybrid **SparseLA-H** architecture, which interleaves self-attention with SparseLA blocks.

![Figure 1: Biological analogy between dendritic computation and linear attention](assets/fig1.png)

Figure 1 summarizes the biological analogy that motivates the work: dendritic
structure and internal neuronal dynamics provide a common view of state
updates in linear attention, linking biological plausibility with efficient
foundation-model design.

The release covers the experiments associated with Figures 5 and 6:

- **Cross-modality token-mixer evaluation.** Figure 5 compares convolution, MLP, self-attention, and vector- and matrix-state variants of HGRN, Mamba, and SparseLA on static images, neuromorphic speech, neuromorphic vision, and associative recall.
- **Language-model scaling and efficiency.** Figure 6 provides
  size-independent architecture implementations, approximately 400M example
  configurations, and selected core modules used for long-context SparseLA-H
  integration.
- **Dendrite-inspired sparse memory.** SparseLA sparsifies memory updates by selecting a subset of branches, reducing computation while retaining memory capacity.
- **Hybrid long-context architecture.** SparseLA-H combines periodic self-attention with SparseLA. This repository releases integration components, not the complete pretraining system or trained model weights.

## Repository map

| Paper component | Code |
| --- | --- |
| Figure 5 overview | [`figure5/`](figure5/) |
| Static image classification, detection, and segmentation | [`figure5/static_image/`](figure5/static_image/) |
| SHD and SSC neuromorphic speech | [`figure5/neuromorphic_speech/`](figure5/neuromorphic_speech/) |
| DVS128 Gesture and DVS128 Gait | [`figure5/neuromorphic_vision/`](figure5/neuromorphic_vision/) |
| Multi-Query Associative Recall (MQAR) | [`figure5/mqar/`](figure5/mqar/) |
| Figure 6 overview | [`figure6/`](figure6/) |
| Size-independent model implementations and example configurations | [`figure6/model_comparison/`](figure6/model_comparison/) |
| Long-context core integrations | [`figure6/long_context/`](figure6/long_context/) |
| vLLM inference integration | [`figure6/long_context/inference/`](figure6/long_context/inference/) |
| Sequence-parallel SparseLA module | [`figure6/long_context/sequence_parallel/`](figure6/long_context/sequence_parallel/) |

## Experimental figures

Figure 5 summarizes the cross-modality comparison and maps directly to
[`figure5/`](figure5/):

![Figure 5: SparseLA across static images, neuromorphic data, and associative recall](assets/fig5.png)

Figure 6 summarizes language-model scaling and long-context efficiency and
maps directly to [`figure6/`](figure6/):

![Figure 6: SparseLA language-model and long-context results](assets/fig6.png)

## Installation

There is no single environment that covers every experiment. The subprojects were developed against different PyTorch, CUDA, Triton, and framework versions; create an isolated environment for each task and follow its README.

General prerequisites are:

- Linux with an NVIDIA GPU for CUDA/Triton kernels;
- a CUDA toolkit compatible with the selected PyTorch build;
- Python and package versions appropriate to the specific experiment;
- sufficient GPU memory and, for distributed examples, a working `torch.distributed` setup.

Do not install all task requirements into one environment: the static-image detection/segmentation stacks, neuromorphic experiments, Zoology/MQAR, model-comparison examples, vLLM integration, and sequence-parallel integration have distinct dependencies.

## Quick start

1. Clone the repository:

   ```bash
   git clone https://github.com/BICLab/Dendritic-Foundation-Model.git
   cd Dendritic-Foundation-Model
   ```

2. Choose one experiment and use its dedicated setup:

   - [Figure 5 task guide](figure5/)
   - [Figure 6 model-comparison guide](figure6/model_comparison/)
   - [Figure 6 long-context integration guide](figure6/long_context/)

3. Prepare the corresponding public dataset and replace any machine-specific paths documented by the selected task.

The repository is organized as a collection of task-specific research stacks.
Each task README documents its data interface, dependencies, launch commands,
and any required integration with an external framework.

## Data availability

All datasets used in the study are publicly available at the links reported in the paper:

- [ImageNet](https://www.image-net.org/)
- [COCO](https://cocodataset.org/)
- [ADE20K](https://ade20k.csail.mit.edu/)
- [SHD and SSC](https://zenkelab.org/datasets/)
- [DVS128 Gesture](https://research.ibm.com/publications/a-low-power-fully-event-based-gesture-recognition-system)
- [DVS128 Gait](https://github.com/zhangxiann/TPAMI_Gait_Identification)
- [MQAR](https://github.com/HazyResearch/zoology)
- [Commonsense Reasoning, SCROLLS, GSM8K, and MATH](https://github.com/EleutherAI/lm-evaluation-harness)
- [RULER](https://github.com/NVIDIA/RULER)

Additional data supporting the results are reported in the paper’s Extended Data Tables. Dataset licenses and access conditions are controlled by the respective providers.

## Code and model availability

The paper’s code-availability URL is <https://github.com/BICLab/Dendritic-Foundation-Model>.

The model-comparison directories contain size-independent architecture code
and example configurations around the 400M-parameter scale. They do **not**
contain released checkpoints. The same implementations can be configured for
the reported 1B- and 3B-parameter experiments by scaling the architecture
fields; training those variants requires the corresponding data and
distributed infrastructure. The long-context directories contain core
integration modules rather than the full 4M/50M-token training and evaluation
stack.

## Acknowledgements and third-party code

The authors thank Prof. Giacomo Indiveri for insightful discussions. Funding acknowledgements are provided in the paper.

This repository incorporates or adapts research code from third-party projects, including Zoology, vLLM, Mamba, Hugging Face Transformers, Flash Linear Attention, MetaFormer-derived vision stacks, and related CUDA/Triton kernels. Retain upstream notices and review [`THIRD_PARTY.md`](THIRD_PARTY.md) and the bundled license texts before redistribution. In particular, selected Meta-derived vision utilities are covered by CC BY-NC 4.0. Those components remain subject to their respective terms.

## Contributing and license

See [`CONTRIBUTING.md`](CONTRIBUTING.md) before submitting changes.

The authors’ original code and documentation are released under the
[Apache License 2.0](LICENSE). Bundled and adapted third-party files remain
governed by their own notices and licenses as documented in
[`THIRD_PARTY.md`](THIRD_PARTY.md).
