# Zoology

This directory contains the Zoology-based experiment framework used for the
Multi-Query Associative Recall evaluation in Figure 5. It is adapted from
[HazyResearch/Zoology](https://github.com/HazyResearch/zoology).

## Installation

Install PyTorch and Transformers for the target CUDA environment, then install
the package in editable mode:

```bash
python -m pip install -e .
```

Optional sweep and analysis dependencies are available through:

```bash
python -m pip install -e ".[extra,analysis]"
```

## SparseLA experiment

The Figure 5 configuration is defined in
[`zoology/model_sp.py`](zoology/model_sp.py). From this directory, run:

```bash
MQAR_MIXERS=sparsela python -m zoology.launch \
  zoology/model_sp.py --gpus 0 --outdir logs
```

See the [MQAR task guide](../README.md) for dataset settings, environment
variables, and the paper protocol.

## Upstream project

Zoology is a framework for evaluating language-model architectures on
synthetic tasks. Upstream documentation and development history are available
in the [original repository](https://github.com/HazyResearch/zoology).

This adapted copy retains the upstream [Apache License 2.0](LICENSE.md).
