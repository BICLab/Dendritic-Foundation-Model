# Figure 5: Static image processing

This directory contains launch templates and operator code for ImageNet-1k
classification, COCO object detection, and ADE20K semantic segmentation.

## Datasets and protocol

| Task | Dataset | Paper training setting | Launch templates |
| --- | --- | --- | --- |
| Classification | [ImageNet-1k](https://www.image-net.org/) | Approximately 25M-parameter pyramid backbone; AdamW; 310 epochs; batch size 256; initial learning rate `5e-4`; cosine decay | [`classification/`](classification/) |
| Detection | [COCO](https://cocodataset.org/) | Fine-tune the pretrained backbone for 12 epochs; batch size 8; initial learning rate `1e-4`; AdamW | [`detection/maskrcnn/`](detection/maskrcnn/) |
| Segmentation | [ADE20K](https://ade20k.csail.mit.edu/) | Fine-tune the pretrained backbone for 64 epochs; batch size 8; initial learning rate `6e-5`; AdamW | [`segmentation/upernet/`](segmentation/upernet/) |

The paper follows a MetaFormer-style pyramid architecture. Pretrain the backbone on ImageNet-1k before using it for the COCO and ADE20K downstream tasks.

## Environment

Use separate environments for classification and the OpenMMLab-based downstream tasks:

```bash
# Classification support packages
python -m pip install -r requirements.txt

# Detection: create a separate environment, then
python -m pip install -r detection/maskrcnn/requirements.txt

# Segmentation: create another separate environment, then
python -m pip install -r segmentation/upernet/requirements.txt
```

Detection pins PyTorch 2.0.1, torchvision 0.15.2, timm 0.5.4,
mmcv-full 1.7.1, and mmdet 2.28.2. Segmentation uses the same base stack with
mmsegmentation 0.30.0. Install `mmcv-full` against a compatible CUDA/PyTorch
build. Classification uses timm 0.9.6 and the listed export/runtime utilities.

The included Mamba and HGRN operators may require compiling CUDA extensions under [`ops/`](ops/). A CUDA-enabled Linux environment and a compatible compiler toolchain are expected.

## Data preparation

- **ImageNet-1k:** arrange the standard `train/` and `val/` class-directory trees and export `DATA_PATH` before running a classification template.
- **COCO:** prepare the standard COCO image and annotation layout expected by MMDetection.
- **ADE20K:** prepare the standard ADE20K layout expected by MMSegmentation.
- **Downstream checkpoints:** update the relevant OpenMMLab configuration with the ImageNet-pretrained backbone checkpoint before fine-tuning.

## Launch templates and path configuration

The repository includes the original shell launch templates:

- Classification examples: [`classification/train_la.sh`](classification/train_la.sh), [`classification/train_mamba_hybrid.sh`](classification/train_mamba_hybrid.sh), and the other `classification/train_*.sh` variants.
- Detection examples: [`detection/maskrcnn/train_metala_tiny.sh`](detection/maskrcnn/train_metala_tiny.sh) and the other `train_*_tiny.sh` variants.
- Segmentation examples: [`segmentation/upernet/train_metala.sh`](segmentation/upernet/train_metala.sh) and the other `train_*.sh` variants.

The launch templates share [`run_env.sh`](run_env.sh). Before launching, set:

- `DATA_PATH` for the target dataset;
- `CUDA_VISIBLE_DEVICES` for the visible GPUs;
- optionally `NPROC_PER_NODE` and `MASTER_PORT`;
- `CHECKPOINT` when using `classification/validate.sh`;
- dataset roots, pretrained checkpoints, work directories, and evaluation settings in the OpenMMLab configurations.

Their environment interface is:

```bash
export DATA_PATH=/path/to/imagenet
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
cd classification
bash train_la.sh
```

The shell files preserve the experiment commands and are intended for use
with the corresponding classification, MMDetection, and MMSegmentation
training stacks.

[Back to Figure 5](../README.md)

