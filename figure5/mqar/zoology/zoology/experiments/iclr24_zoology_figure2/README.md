# Reproducing Zoology Figure 2

This experiment accompanies *[Zoology: Measuring and Improving Recall in Efficient Language Models](https://arxiv.org/abs/2312.04927)* and its [analysis post](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology1-analysis).

The x-axis is model dimension and the y-axis is MQAR accuracy. Increasing sequence length increases task difficulty; each result is the maximum over four learning rates.

From the Zoology package root, configure W&B and run:

```bash
python -m zoology.launch zoology/experiments/iclr24_zoology_figure2/configs.py -p
```

The full sweep contains 448 model/data configurations. The upstream
experiments used eight A100 GPUs with `-p`. For a smaller run, reduce the
loops in [`configs.py`](configs.py).

Plotting implementation:
[`zoology/analysis/paper/zoology_figure2.py`](../../analysis/paper/zoology_figure2.py).