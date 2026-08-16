# Figure 5: Cross-modality token-mixer benchmarks

Figure 5 evaluates nine token mixers across four modality-specific settings: convolution, MLP, self-attention, and 1D/vector and 2D/matrix variants of HGRN, Mamba, and SparseLA.

## Tasks

| Task | Dataset(s) | Entry points |
| --- | --- | --- |
| [Static image](static_image/) | ImageNet-1k, COCO, ADE20K | Shell launch templates under `classification/`, `detection/maskrcnn/`, and `segmentation/upernet/` |
| [Neuromorphic speech](neuromorphic_speech/) | SHD, SSC | `shd_generate_dataset.py`, `ssc_generate_dataset.py`, `shd_train.py`, `ssc_train.py` |
| [Neuromorphic vision](neuromorphic_vision/) | DVS128 Gesture, DVS128 Gait | `main_finetune.py`, `train_gesture.sh`, `train_gait_day.sh` |
| [Associative recall](mqar/) | Synthetic MQAR | `zoology/zoology/model_sp.py` through `python -m zoology.launch` |

## Paper protocols

- **Static images:** pretrain an approximately 25M-parameter pyramid backbone on ImageNet-1k, then fine-tune on COCO and ADE20K. Classification uses AdamW for 310 epochs, batch size 256, and initial learning rate `5e-4`; detection uses 12 epochs, batch size 8, and `1e-4`; segmentation uses 64 epochs, batch size 8, and `6e-5`.
- **Neuromorphic speech:** convert each one-second spike stream to 250 frames at `dt = 4 ms`, producing `250 × 700` inputs. Train approximately 0.3M-parameter models for 200 epochs with AdamW, cosine decay, weight decay `0.1`, and an initial learning rate of `1e-2` or `5e-3`. Results use five runs.
- **Neuromorphic vision:** use 36 event frames with `dt = 15 ms` at the original `128 × 128` spatial resolution. Train approximately 0.5M-parameter models for 200 epochs with Lamb, 10 warm-up epochs to `3e-4`, cosine decay to `1e-5`, and weight decay `6e-2`. Results use five runs.
- **MQAR:** generate 100,000 training and 3,000 test samples. Use two-layer models with dimensions 32, 64, 128, 256, and 512; sweep four log-spaced learning rates from `1e-4` to `1e-2`; and train for 64 epochs. The paper evaluates sequence length/KV-pair settings `256/16`, `512/64`, and `1024/128`, with batch sizes 256, 128, and 64.

## Environment and paths

Each task uses its own dependencies and data layout:

- [Static image setup](static_image/README.md)
- [Neuromorphic speech setup](neuromorphic_speech/README.md)
- [Neuromorphic vision setup](neuromorphic_vision/README.md)
- [MQAR setup](mqar/README.md)

Dataset roots, GPU visibility, output directories, and experiment selections
are configurable through the command-line arguments and environment variables
documented by each task.

[Back to repository overview](../README.md)
