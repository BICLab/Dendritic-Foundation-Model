# Figure 5: Multi-Query Associative Recall

This task evaluates token mixers on Multi-Query Associative Recall (MQAR), using a modified copy of [HazyResearch/Zoology](https://github.com/HazyResearch/zoology).

## Dataset and paper protocol

MQAR is generated synthetically by [`zoology/data/associative_recall.py`](zoology/zoology/data/associative_recall.py); no external dataset download is required. The paper uses:

- 100,000 training examples and 3,000 test examples;
- vocabulary size 8,192;
- two model layers;
- model dimensions 32, 64, 128, 256, and 512;
- four log-spaced learning rates from `1e-4` to `1e-2`;
- 64 epochs;
- sequence length/KV-pair settings `256/16`, `512/64`, and `1024/128`;
- corresponding batch sizes 256, 128, and 64.

Generated data are cached according to `DataConfig.cache_dir`.

## Environment

From the vendored Zoology root, install PyTorch and Transformers first, then install the package:

```bash
cd zoology
python -m pip install -e .
```

The package declares NumPy, einops, tqdm, click, Pydantic `>=2.0.0,<2.5.0`, W&B, `mamba-ssm`, and `causal-conv1d`. Install optional sweep/analysis dependencies with:

```bash
python -m pip install -e ".[extra,analysis]"
```

CUDA mixer implementations additionally require mutually compatible PyTorch, CUDA, Triton, Flash Linear Attention (`fla`), and compiled extension versions. See the [vendored Zoology guide](zoology/README.md) and its [license](zoology/LICENSE.md).

## Entry point

The released experiment definition is [`zoology/model_sp.py`](zoology/zoology/model_sp.py), launched through [`zoology.launch`](zoology/zoology/launch.py):

```bash
cd zoology
mkdir -p logs
MQAR_MIXERS=sparsela python -m zoology.launch \
  zoology/model_sp.py --gpus 0 --outdir logs
```

The launcher also supports `--outdir`, `--name`, `--gpus`, and
`--parallelize`/`-p`; parallel mode requires Ray. `--gpus` sets
`CUDA_VISIBLE_DEVICES`. Set `MQAR_MIXERS` to a comma-separated subset of the
mixers configured in `model_sp.py`; the default is `sparsela`. Optional
runtime settings are `ZOOLOGY_CACHE_DIR`, `WANDB_PROJECT`, and
`WANDB_ENTITY`.

[`zoology/run.sh`](zoology/run.sh) is a foreground convenience wrapper:

```bash
CUDA_VISIBLE_DEVICES=0 MQAR_MIXERS=sparsela bash zoology/run.sh
```

Set `RUN_IN_BACKGROUND=1` only when asynchronous execution is desired.

## Release scope

The checked-in `model_sp.py` is a reduced example: it currently enables only
the `512/64` data setting, model dimension 64, and one learning-rate
selection. Several paper sweep values are commented out. The default selects
only the released SparseLA mixer. Some baseline entries reference mixer
modules that are not included in this release, so selecting every configured
name does not constitute a complete Figure 5 reproduction.

Accordingly, the provided command launches the released SparseLA setting,
not every Figure 5 MQAR point. Use the paper settings above to construct the
complete sweep.

[Back to Figure 5](../README.md)