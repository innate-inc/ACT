# ACT

[Action Chunking Transformer](https://arxiv.org/abs/2304.13705) for robot imitation learning.

In production this repo is downloaded by the [innate training orchestrator](https://github.com/innate-inc/innate-cloud/tree/main/apps/training-orchestrator), which launches a GPU instance and runs `cloud_run.sh`. See env vars in that script for configuration.

To test locally on a machine with GPUs:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .
PYTHONPATH=act_test:$PYTHONPATH python -m act_test.train_dist --data_dir /path/to/hdf5_data --world_size 1
```

## `train_dist.py` args

| Arg | Default | Description |
|---|---|---|
| `--data_dir` | (required) | Path to HDF5 dataset directory |
| `--world_size` | all GPUs | Number of GPUs |
| `--batch_size` | 96 | Batch size per GPU |
| `--chunk_size` | 30 | Action sequence length |
| `--max_steps` | 120000 | Training steps |
| `--learning_rate` | 5e-5 | Learning rate |
| `--learning_rate_backbone` | 5e-5 | Vision backbone learning rate |
| `--warmup_steps` | 5% of max_steps | LR warmup steps |
| `--num_workers` | 4 | DataLoader workers per GPU |
| `--shard_size` | 500 | Samples per WebDataset shard |
| `--force-reconvert` | off | Reconvert HDF5→WebDataset even if shards exist |
| `--no-bf16` | bf16 on | Disable BF16 mixed precision |
| `--no-compile` | compile on | Disable `torch.compile()` |

When run via `cloud_run.sh` (production), these are set from env vars: `DATA_DIR`, `OUTPUT_DIR`, `WORLD_SIZE`, `CHUNK_SIZE`, `MAX_STEPS`, `LEARNING_RATE`, `LEARNING_RATE_BACKBONE`, `BATCH_SIZE`, `NUM_WORKERS`.
