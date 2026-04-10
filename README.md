# ACT – Action Chunking Transformer

[Action Chunking Transformer](https://arxiv.org/abs/2304.13705) implementation for robot imitation learning. Predicts chunks of future actions from dual-camera RGB observations and proprioceptive state.

## How This Repo Is Used

**In production**, this repo is cloned automatically by the [innate training orchestrator](https://github.com/innate-inc/innate-cloud/tree/main/apps/training-orchestrator). The orchestrator launches a GPU instance, downloads this repo + training data, and runs `cloud_run.sh` as the entrypoint. You don't need to manually set anything up for production training.

**For local testing**, you can run the training script directly on a machine with GPUs:

```bash
# Install
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .

# Run training (auto-converts HDF5 → WebDataset, then trains with DDP)
PYTHONPATH=act_test:$PYTHONPATH python -m act_test.train_dist \
    --data_dir /path/to/hdf5_data \
    --world_size 1
```

## Entrypoints

| Entrypoint | Used By | Description |
|---|---|---|
| `cloud_run.sh` | Training orchestrator | Installs deps, detects GPU arch, runs `train_dist.py`. Configured via env vars (`DATA_DIR`, `WORLD_SIZE`, `MAX_STEPS`, etc.) |
| `python -m act_test.train_dist` | Local testing / `cloud_run.sh` | DDP training: HDF5→WebDataset conversion, multi-GPU training, checkpointing, ONNX export |

## Data Format

Training data is HDF5 episode files + a `metadata.json`:

```
dataset_directory/
├── metadata.json
├── episode_0.h5
├── episode_1.h5
└── ...
```

Each HDF5 file contains:
- `action` — `(T, 10)` action vectors
- `observations/qpos` — `(T, 6)` joint positions
- `observations/images/camera_1` — `(T, H, W, 3)` RGB images
- `observations/images/camera_2` — `(T, H, W, 3)` RGB images

`train_dist.py` automatically converts HDF5 to [WebDataset](https://github.com/webdataset/webdataset) tar shards before training.

## Environment Variables (`cloud_run.sh`)

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `/training/data/data` | Path to HDF5 training data |
| `OUTPUT_DIR` | `/training/out` | Checkpoint output directory |
| `WORLD_SIZE` | `4` | Number of GPUs |
| `MAX_STEPS` | `120000` | Training steps |
| `BATCH_SIZE` | `96` | Batch size per GPU |
| `LEARNING_RATE` | `5e-5` | Learning rate |
| `LEARNING_RATE_BACKBONE` | `5e-5` | Backbone learning rate |
| `CHUNK_SIZE` | `30` | Action sequence length |
| `NUM_WORKERS` | `4` | DataLoader workers per GPU |
