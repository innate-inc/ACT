# ACT – Action Chunking Transformer

An implementation of the **Action Chunking Transformer (ACT)** policy for robot imitation learning, based on the paper [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705). ACT predicts a *chunk* of future actions from visual observations and proprioceptive state, enabling smooth and precise robotic manipulation.

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Format](#data-format)
  - [HDF5 Episode Format](#hdf5-episode-format)
  - [WebDataset Format](#webdataset-format)
  - [Data Conversion](#data-conversion)
- [Training](#training)
  - [Single-GPU Training](#single-gpu-training)
  - [Distributed Training (Multi-GPU)](#distributed-training-multi-gpu)
  - [Cloud / Docker Training](#cloud--docker-training)
- [Inference & Benchmarks](#inference--benchmarks)
- [Configuration Reference](#configuration-reference)
- [Project Modules](#project-modules)

---

## Features

| Capability | Details |
|---|---|
| **Action Chunking** | Predicts sequences of actions (default 30 steps) rather than single actions |
| **VAE Latent Space** | Optional variational autoencoder for multi-modal action distributions |
| **Multi-Camera Input** | Dual-camera RGB observations (camera\_1, camera\_2) |
| **WebDataset Streaming** | Scalable streaming data pipeline via WebDataset tar shards |
| **Distributed Training** | Multi-GPU training with PyTorch DDP |
| **Mixed Precision** | BF16 training support for faster throughput |
| **torch.compile** | Optional Inductor compilation for optimized training |
| **ONNX Export** | Automatic ONNX export at end of distributed training |
| **TensorRT Benchmark** | Benchmark script comparing PyTorch vs TensorRT inference |
| **Learning Rate Schedule** | Linear warmup + cosine annealing |
| **W&B Logging** | Weights & Biases integration for experiment tracking |

---

## Repository Structure

```
ACT/
├── act_test/                   # Main Python package
│   ├── __init__.py
│   ├── ACT.py                  # Core model: ACTConfig, ACTPolicy, transformer layers
│   ├── data_utils.py           # Data loading: HDF5 dataset, WebDataset pipelines, stats
│   ├── train.py                # Single-GPU training script
│   ├── train_dist.py           # Multi-GPU distributed training script (DDP)
│   ├── test.py                 # End-to-end data conversion pipeline test
│   ├── compute_benchmark.py    # Forward/backward pass GPU benchmark
│   ├── inference_benchmark.py  # Inference latency benchmark
│   ├── dataloading_benchmark.py# Data loading throughput benchmark
│   ├── tensort_benchmark.py    # TensorRT vs PyTorch inference benchmark
│   └── data_tools/
│       └── webdataset.py       # HDF5 → WebDataset conversion utilities
├── cloud_run.sh                # Docker entrypoint: env setup + distributed training
├── create_raid.sh              # NVMe RAID-0 setup for cloud GPU instances
├── Dockerfile                  # CUDA 12.8 container image
├── setup.py                    # Package definition and dependencies
├── requirements.txt            # Pinned dependency versions
└── manifest.in                 # Package data manifest
```

---

## Architecture Overview

ACT follows an encoder-decoder transformer architecture with an optional VAE:

```
Observations                      Actions (training only)
  ├── Camera 1 image ─┐              │
  ├── Camera 2 image ─┤              ▼
  └── Robot state ─────┤        ┌──────────┐
                       │        │ VAE Enc. │ ← Encodes (state + actions) → μ, σ
                       │        └────┬─────┘
                       │             │ z ~ N(μ, σ)   (or z = 0 at inference)
                       │             │
                       ▼             ▼
                 ┌───────────────────────┐
                 │    Main Encoder       │ ← ResNet18 backbone + 1D projections
                 │  (image features +    │   + sinusoidal 2D positional embeddings
                 │   state + latent z)   │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │    Main Decoder       │ ← Learnable action query embeddings
                 │  (cross-attention to  │
                 │   encoder output)     │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │   Action Head         │ ← Linear projection → (chunk_size, action_dim)
                 └───────────────────────┘
```

**Key components** (all in `act_test/ACT.py`):

| Class | Purpose |
|---|---|
| `ACTConfig` | Dataclass holding all model hyperparameters |
| `ACTPolicy` | Top-level policy: normalization, forward/loss, `select_action` for inference |
| `ACT` | Core transformer: VAE encoder, ResNet backbone, main encoder/decoder, action head |
| `ACTEncoder` / `ACTDecoder` | Stacked transformer encoder/decoder layers |
| `ACTTemporalEnsembler` | Exponential-weighted temporal ensembling of action chunks |
| `Normalize` / `Unnormalize` | Per-feature normalization using dataset statistics |

---

## Prerequisites

- **Python** ≥ 3.10
- **CUDA**-capable GPU (tested with NVIDIA A100, B200)
- **PyTorch** 2.6+ with matching CUDA toolkit

---

## Installation

### Local Setup

```bash
# Clone the repository
git clone https://github.com/innate-inc/ACT.git
cd ACT

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121

# Install the package and remaining dependencies
pip install -e .
```

### Docker

```bash
docker build -t act-training .
```

The Dockerfile uses `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04` as the base. Python environment setup and PyTorch installation happen at runtime via `cloud_run.sh` to support GPU-architecture-specific builds (e.g., Blackwell nightly wheels).

---

## Data Format

### HDF5 Episode Format

Training data is organized as individual HDF5 episode files alongside a metadata JSON:

```
dataset_directory/
├── metadata.json          # (or dataset_metadata.json)
├── episode_0.h5
├── episode_1.h5
└── ...
```

**`metadata.json`** structure:
```json
{
  "task_name": "MyTask",
  "episodes": [
    {"episode_id": 0, "file_name": "episode_0.h5"},
    {"episode_id": 1, "file_name": "episode_1.h5"}
  ]
}
```

**Each HDF5 file** contains:
| Dataset Key | Shape | Description |
|---|---|---|
| `action` | `(T, 10)` | Action vectors (e.g., joint velocities + gripper) |
| `observations/qpos` | `(T, 6)` | Joint positions (proprioceptive state) |
| `observations/images/camera_1` | `(T, H, W, 3)` | RGB images from camera 1 |
| `observations/images/camera_2` | `(T, H, W, 3)` | RGB images from camera 2 |

where `T` is the number of timesteps in the episode.

### WebDataset Format

For efficient streaming training, HDF5 data is converted to [WebDataset](https://github.com/webdataset/webdataset) tar shards:

```
webdataset_directory/
├── train-00000.tar
├── train-00001.tar
├── ...
└── dataset_info.json
```

Each tar shard contains samples with four files per timestep:

| File | Format | Content |
|---|---|---|
| `{key}.cam1.pth` | `torch.uint8` tensor `(224, 224, 3)` | Resized camera 1 image (HWC) |
| `{key}.cam2.pth` | `torch.uint8` tensor `(224, 224, 3)` | Resized camera 2 image (HWC) |
| `{key}.qpos.pth` | `torch.float16` tensor `(6,)` | Joint positions |
| `{key}.actions.pth` | `torch.float16` tensor `(remaining_T, 10)` | Future actions from this timestep |

### Data Conversion

Convert HDF5 episodes to WebDataset format:

```python
from act_test.data_tools.webdataset import convert_hdf5_to_webdataset

convert_hdf5_to_webdataset(
    hdf5_directory="/path/to/hdf5_data",
    webd_directory="/path/to/output_webdataset",
    shard_size=1000,        # samples per shard
    target_size=(224, 224)  # resize images to 224×224
)
```

Or test the full conversion + loading pipeline:

```bash
python -m act_test.test --local-dir /path/to/hdf5_data --output-dir /path/to/webdataset
```

> **Note:** Distributed training (`train_dist.py`) and single-GPU training (`train.py`) automatically convert HDF5 data to WebDataset format before training begins.

---

## Training

### Single-GPU Training

Edit the configuration constants at the top of `act_test/train.py` (data paths, hyperparameters, W&B settings), then:

```bash
export PYTHONPATH=act_test:$PYTHONPATH
python act_test/train.py
```

Key configuration variables in `train.py`:

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | — | Path to HDF5 dataset directory |
| `CHUNK_SIZE` | 30 | Action sequence prediction length |
| `BATCH_SIZE` | 96 | Training batch size |
| `MAX_STEPS` | 60000 | Total training steps |
| `LEARNING_RATE` | 1e-5 | Learning rate |
| `USE_VAE` | True | Enable VAE latent space |
| `KL_WEIGHT` | 10.0 | KL divergence loss weight |

### Distributed Training (Multi-GPU)

The recommended training script for production use:

```bash
export PYTHONPATH=act_test:$PYTHONPATH

python -m act_test.train_dist \
    --data_dir /path/to/hdf5_data \
    --chunk_size 30 \
    --max_steps 120000 \
    --learning_rate 5e-5 \
    --learning_rate_backbone 5e-5 \
    --batch_size 96 \
    --num_workers 4 \
    --world_size 4
```

Additional optimization flags:

| Flag | Default | Description |
|---|---|---|
| `--use-bf16` / `--no-bf16` | Enabled | BF16 mixed-precision training |
| `--use-compile` / `--no-compile` | Enabled | `torch.compile()` with Inductor |
| `--warmup_steps` | 5% of `max_steps` | Linear LR warmup duration |
| `--shard_size` | 500 | WebDataset samples per shard |
| `--force-reconvert` | Off | Reconvert HDF5→WebDataset even if shards exist |

**Training pipeline:**
1. Converts HDF5 data → WebDataset shards (once, before spawning workers)
2. Spawns `world_size` processes via `torch.multiprocessing.spawn`
3. Each GPU runs DDP training with its own data stream
4. Checkpoints saved every `max_steps / 10` steps
5. ONNX model exported at end of training

**Learning rate schedule:** Linear warmup (default 5% of steps) → cosine annealing to 10% of peak LR.

### Cloud / Docker Training

```bash
docker run --gpus all \
    -v /path/to/data:/training/data \
    -v /path/to/output:/training/out \
    -e DATA_DIR=/training/data/data \
    -e OUTPUT_DIR=/training/out \
    -e WORLD_SIZE=4 \
    -e WANDB_API_KEY=your_key \
    act-training
```

`cloud_run.sh` handles:
- GPU detection and architecture-specific PyTorch installation (including Blackwell nightly builds)
- Virtual environment creation with `uv`
- Dependency installation
- Launching `train_dist.py` with environment variable configuration

For cloud instances with local NVMe SSDs, use `create_raid.sh` to set up a RAID-0 array first:

```bash
sudo ./create_raid.sh    # Creates /home/user/raid mount point
```

---

## Inference & Benchmarks

### Action Selection (Inference)

```python
import torch
from act_test.ACT import ACTConfig, ACTPolicy

# Load config and model
config = ACTConfig(
    input_shapes={
        "observation.image_camera_1": [3, 224, 224],
        "observation.image_camera_2": [3, 224, 224],
        "observation.state": [6],
    },
    output_shapes={"action": [10]},
    chunk_size=30,
    n_action_steps=30,
)

# Load dataset stats and trained weights
dataset_stats = torch.load("checkpoint_dir/dataset_stats.pt")
policy = ACTPolicy(config=config, dataset_stats=dataset_stats)
policy.load_state_dict(torch.load("checkpoint_dir/act_policy_step_60000.pth"))
policy.eval().cuda()

# Run inference
batch = {
    "observation.image_camera_1": cam1_tensor,  # (1, 3, 224, 224)
    "observation.image_camera_2": cam2_tensor,  # (1, 3, 224, 224)
    "observation.state": qpos_tensor,           # (1, 6)
}
action = policy.select_action(batch)  # (1, 10)
```

`select_action` maintains an internal action queue of length `n_action_steps`. A new forward pass is triggered only when the queue is empty. For temporal ensembling, set `temporal_ensemble_coeff` in the config.

### Benchmark Scripts

All benchmarks are in `act_test/` and intended to run on a GPU:

| Script | What it Measures |
|---|---|
| `compute_benchmark.py` | Forward + backward pass timing and GPU memory with synthetic data |
| `inference_benchmark.py` | Inference latency (forward pass only, `torch.compile` enabled) |
| `dataloading_benchmark.py` | WebDataset data loading throughput across workers/GPUs |
| `tensort_benchmark.py` | TensorRT vs PyTorch inference comparison (requires ONNX checkpoint) |

---

## Configuration Reference

### `ACTConfig` Fields

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_obs_steps` | int | 1 | Number of observation timesteps (currently fixed at 1) |
| `chunk_size` | int | 100 | Transformer context length / action chunk size |
| `n_action_steps` | int | 100 | Number of action steps to return from the queue |
| `speed` | float | 1.0 | Action resampling speed factor |
| `input_shapes` | dict | `{}` | Map of input names → shapes (e.g., image, state) |
| `output_shapes` | dict | `{}` | Map of output names → shapes (e.g., action) |
| `vision_backbone` | str | `"resnet18"` | ResNet variant for image encoding |
| `pretrained_backbone_weights` | str | `"ResNet18_Weights.IMAGENET1K_V1"` | Pretrained weights for the backbone |
| `dim_model` | int | 512 | Transformer hidden dimension |
| `n_heads` | int | 8 | Number of attention heads |
| `dim_feedforward` | int | 3200 | Feed-forward network dimension |
| `n_encoder_layers` | int | 4 | Main encoder transformer layers |
| `n_decoder_layers` | int | 1 | Main decoder transformer layers |
| `use_vae` | bool | True | Enable VAE encoder during training |
| `latent_dim` | int | 32 | VAE latent space dimension |
| `n_vae_encoder_layers` | int | 4 | VAE encoder transformer layers |
| `kl_weight` | float | 10.0 | Weight of KL divergence loss |
| `dropout` | float | 0.1 | Transformer dropout rate |
| `temporal_ensemble_coeff` | float | None | Temporal ensembling coefficient (None = disabled) |

### Environment Variables (`cloud_run.sh`)

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `/training/data/data` | Path to training data |
| `OUTPUT_DIR` | `/training/out` | Path for checkpoints and outputs |
| `WORLD_SIZE` | 4 | Number of GPUs |
| `CHUNK_SIZE` | 30 | Action sequence length |
| `MAX_STEPS` | 120000 | Maximum training steps |
| `LEARNING_RATE` | 5e-5 | Main learning rate |
| `LEARNING_RATE_BACKBONE` | 5e-5 | Backbone learning rate |
| `BATCH_SIZE` | 96 | Batch size per GPU |
| `NUM_WORKERS` | 4 | DataLoader workers per GPU |

---

## Project Modules

### `act_test/ACT.py`

Core model implementation (~1035 lines). Contains:
- **Normalization utilities**: `Normalize`, `Unnormalize`, `NormalizationMode` — per-feature mean/std or min/max normalization using dataset statistics stored as model buffers
- **`ACTConfig`** — dataclass with all model hyperparameters and helper properties for feature typing
- **`ACTPolicy`** — top-level `nn.Module` that wraps normalization, the transformer model, and provides `forward()` (training with loss) and `select_action()` (inference with action queue)
- **`ACT`** — the core transformer: optional VAE encoder, ResNet18 backbone with frozen batch norm, main encoder/decoder with sinusoidal positional embeddings, and a linear action regression head
- **`ACTTemporalEnsembler`** — implements Algorithm 2 from the ACT paper for smooth action execution

### `act_test/data_utils.py`

Data loading and statistics (~665 lines). Contains:
- **`EpisodicHDF5DatasetRAM`** — loads all HDF5 episodes into RAM, computes normalization statistics, samples random timesteps with action chunking and padding
- **`WebDatasetStreaming`** / **`WebDatasetDecoder`** — streaming WebDataset pipeline that decodes `.pth` tensors, handles image format conversion (uint8 HWC → float32 CHW), and action padding
- **`calculate_webdataset_stats()`** — online computation of per-feature mean/std from a streaming dataloader
- **`initialize_webdataset_data()`** — creates train/val WebDataset dataloaders with random splitting and automatic statistics computation

### `act_test/data_tools/webdataset.py`

HDF5 → WebDataset conversion (~272 lines). Reads HDF5 episodes, resizes images to 224×224, stores each timestep as a set of `.pth` tensor files packed into tar shards.

### `act_test/train_dist.py`

Production distributed training script (~721 lines). Handles:
- HDF5 → WebDataset conversion (before DDP spawn)
- `torch.multiprocessing.spawn` based DDP setup
- BF16 mixed precision with `torch.amp.autocast`
- `torch.compile()` with Inductor backend
- Linear warmup + cosine annealing LR schedule
- Periodic validation, checkpointing, W&B logging
- ONNX export at training completion

---

## License

See [LICENSE](LICENSE) for details.
