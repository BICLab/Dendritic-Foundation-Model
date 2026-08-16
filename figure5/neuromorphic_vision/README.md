# Figure 5: Neuromorphic vision recognition

This directory contains event-based recognition models and training entry points for DVS128 Gesture and DVS128 Gait.

## Datasets and protocol

- [DVS128 Gesture](https://research.ibm.com/publications/a-low-power-fully-event-based-gesture-recognition-system): 1,342 samples, 11 gestures, 29 subjects, and three illumination conditions.
- [DVS128 Gait](https://github.com/zhangxiann/TPAMI_Gait_Identification): 4,200 samples from 21 participants and two viewpoints.

The paper uses frame-based preprocessing at the original `128 × 128` resolution. Each sample contains 36 segments with `dt = 15 ms`, and random consecutive slicing is used for training augmentation. Models are kept near 0.5M parameters and evaluated over five runs. The reported training protocol is 200 epochs with Lamb, 10 linear warm-up epochs to `3e-4`, cosine decay to `1e-5`, and weight decay `6e-2`.

The provided launch scripts pass the reported epoch, frame, warm-up, minimum-learning-rate, and weight-decay values. [`main_finetune.py`](main_finetune.py) constructs timm’s Lamb optimizer.

## Environment

No lockfile is included for this task. Create a dedicated CUDA environment and install compatible versions of:

- PyTorch, torchvision, NumPy, pandas, h5py, Pillow, and TensorBoard;
- timm, torchinfo, einops, and spikingjelly;
- Flash Linear Attention (`fla`), Triton, `causal-conv1d`, and `mamba-ssm` for the corresponding token mixers;
- a compiler/CUDA toolkit for the HGRN2 extension under [`hgru2_pytorch/`](hgru2_pytorch/).

The code imports the legacy `spikingjelly.clock_driven` API, so a recent incompatible spikingjelly release may require adaptation.

## Data preparation

Pass dataset roots through `--data_path`:

- Gesture expects `<data_path>/train/` and `<data_path>/test/`, each containing per-sample HDF5 files with `times`, `addrs`, and `labels` datasets.
- Gait expects:

  ```text
  <data_path>/
  ├── train/
  │   ├── train_data.npy
  │   └── train_target.npy
  └── test/
      ├── test_data.npy
      └── test_target.npy
  ```

The training loaders consume these preprocessed layouts.

## Entry points

[`main_finetune.py`](main_finetune.py) is the Python entry point. Run one model at a time after replacing the dataset and output paths:

```bash
CUDA_VISIBLE_DEVICES=0 python main_finetune.py \
  --dataset gesture \
  --data_path /path/to/DVSGesture_data \
  --model V3_tem_attn_sparsela_m_tiny \
  --model_mode tem_attn \
  --time_steps 36 \
  --batch_size 64 --batch_size_val 8 \
  --epochs 200 --warmup_epochs 10 \
  --lr 3e-4 --min_lr 1e-5 --weight_decay 6e-2 \
  --output_dir ./outputs/sparsela_m_gesture \
  --log_dir ./outputs/sparsela_m_gesture
```

For gait, use `--dataset gait-day`, the converted gait root, and the paper/script batch settings:

```bash
CUDA_VISIBLE_DEVICES=0 python main_finetune.py \
  --dataset gait-day \
  --data_path /path/to/dvs-gait/npy \
  --model V3_tem_attn_sparsela_m_tiny \
  --model_mode tem_attn \
  --time_steps 36 \
  --batch_size 32 --batch_size_val 16 \
  --epochs 200 --warmup_epochs 10 \
  --lr 3e-4 --min_lr 1e-5 --weight_decay 6e-2 \
  --output_dir ./outputs/sparsela_m_gait_day \
  --log_dir ./outputs/sparsela_m_gait_day
```

[`train_gesture.sh`](train_gesture.sh) and
[`train_gait_day.sh`](train_gait_day.sh) run the mixer variants sequentially.
Configure them through `DATA_PATH`, `CUDA_VISIBLE_DEVICES`, `OUTPUT_DIR`, and
the space-separated `MODELS` list:

```bash
DATA_PATH=/path/to/DVSGesture_data \
CUDA_VISIBLE_DEVICES=0 \
MODELS="sparsela_v sparsela_m" \
bash train_gesture.sh
```

## Configurable paths and environment

- CLI paths: `--data_path`, `--output_dir`, `--log_dir`, `--resume`, and `--finetune`.
- CLI runtime controls: `--device`, `--num_workers`, `--seed`, `--world_size`, `--local-rank`, and `--dist_url`.
- Distributed environment: `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, and `MASTER_PORT` are read when using `env://`; SLURM and OpenMPI variables are also recognized.
- Logging: `LOGLEVEL` is read by helper modules.

GPU visibility is controlled by the launch environment. Legacy factory names
`V3_tem_attn_spla1d_tiny` and `V3_tem_attn_spla2d_tiny` map to
`sparsela_v` and `sparsela_m`.

[Back to Figure 5](../README.md)