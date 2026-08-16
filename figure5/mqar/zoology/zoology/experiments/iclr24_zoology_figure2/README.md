# Reproducing Zoology Figure 2

This experiment accompanies *[Zoology: Measuring and Improving Recall in Efficient Language Models](https://arxiv.org/abs/2312.04927)* and its [analysis post](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology1-analysis).

The x-axis is model dimension and the y-axis is MQAR accuracy. Increasing sequence length increases task difficulty; each result is the maximum over four learning rates.

From the vendored Zoology root, configure W&B and run the checked-in experiment file:

```bash
python -m zoology.launch zoology/experiments/iclr24_zoology_figure2/configs.py -p
```

The full sweep contains 448 model/data configurations. The upstream experiments used eight A100 GPUs with `-p`. For a smaller run, reduce the loops in [`configs.py`](configs.py). See the parent [configuration guide](../../../README.md#configuration-experiments-and-sweeps) for launcher details.

The plotting implementation included in this release is
[`zoology/analysis/paper/zoology_figure2.py`](../../analysis/paper/zoology_figure2.py).
The original rendered figure asset is not bundled.