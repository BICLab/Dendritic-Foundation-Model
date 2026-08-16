# Figure 5: Neuromorphic speech recognition

This directory contains the SHD and SSC preprocessing and training entry points used for the neuromorphic speech experiments.

## Datasets and representation

- [Spiking Heidelberg Digits (SHD)](https://zenkelab.org/datasets/) contains 20 spoken-digit classes in German and English.
- [Spiking Speech Commands (SSC)](https://zenkelab.org/datasets/) contains 35 word classes.

The paper aligns each event stream to one second and bins events into 250 frames at `dt = 4 ms`, yielding an input of shape `250 × 700`.

## Environment

Create a dedicated CUDA environment. The source imports PyTorch, torchvision, NumPy, PyTables, Matplotlib, tqdm, Weights & Biases, einops, Flash Linear Attention (`fla`), `causal-conv1d`, `mamba-ssm`, and HGRN2 CUDA modules. Versions are not pinned in this directory, so select mutually compatible PyTorch/CUDA/Triton builds and install PyTorch first.

The training programs use offline W&B logging. A network login is not required for the default `mode="offline"`, but the `wandb` package is required.

## Data preparation

Download the provider's HDF5 files, then run:

```bash
python shd_generate_dataset.py \
  --input-dir /path/to/raw/shd \
  --output-dir ./data/shd

python ssc_generate_dataset.py \
  --input-dir /path/to/raw/ssc \
  --output-dir ./data/ssc
```

The SHD input directory must contain `shd_train.h5` and `shd_test.h5`.
The SSC input directory must contain `ssc_train.h5`, `ssc_valid.h5`, and
`ssc_test.h5`. Both converters default to 4 ms time bins and create their
output directories automatically.

## Training protocol and entry points

The paper trains an MLP encoder followed by two meta-blocks, keeps models near 0.3M parameters, and reports five runs. The common protocol is 200 epochs, AdamW, cosine learning-rate decay, weight decay `0.1`, and an initial learning rate selected from `1e-2` or `5e-3`.

The actual Python entry files are:

- [`shd_train.py`](shd_train.py) for SHD;
- [`ssc_train.py`](ssc_train.py) for SSC.

Representative invocations are:

```bash
CUDA_VISIBLE_DEVICES=0 python shd_train.py \
  --data-path ./data/shd \
  --block sparsela_m --seed 42 --epochs 200 \
  --learning_rate 5e-3 --weight_decay 0.1

CUDA_VISIBLE_DEVICES=0 python ssc_train.py \
  --data-path ./data/ssc \
  --block sparsela_m --seed 42 --epochs 200 \
  --learning_rate 1e-2 --weight_decay 0.1
```

Supported command-line controls include `--data-path`, `--channel_size`,
`--seed`, `--batch_size`, `--optimizer`, `--scheduler`, `--learning_rate`,
`--weight_decay`, `--epochs`, and `--block`. Public SparseLA block names are
`sparsela_v` and `sparsela_m`; historical `spla1d` and `spla2d` names remain
accepted for compatibility.

The convenience wrappers run five seeds and create their log directories:

```bash
DATA_PATH=./data/ssc CUDA_VISIBLE_DEVICES=0 bash run.sh
DATA_PATH=./data/shd CUDA_VISIBLE_DEVICES=0 bash run_shd.sh
```

Override `BLOCK`, `SEEDS`, `OUTPUT_DIR`, or `RUN_IN_BACKGROUND=1` as needed.

The required HGRN implementation is bundled under
[`hgru2_pytorch/`](hgru2_pytorch/). The SHD trainer retains its historical
300-epoch default; pass `--epochs 200` for the paper protocol.

[Back to Figure 5](../README.md)