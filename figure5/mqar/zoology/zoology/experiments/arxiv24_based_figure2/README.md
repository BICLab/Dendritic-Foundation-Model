# Reproducing Based Figure 2

This experiment accompanies [*Simple linear attention language models balance the recall-throughput tradeoff*](https://arxiv.org/abs/2402.18668).

From the vendored Zoology root, configure W&B and run:

```bash
python -m zoology.launch zoology/experiments/arxiv24_based_figure2/configs.py -p
```

The checked-in experiment definition is [`configs.py`](configs.py). The
original rendered `figure.png` is not part of this release.
