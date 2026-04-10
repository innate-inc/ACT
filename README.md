# ACT

[Action Chunking Transformer](https://arxiv.org/abs/2304.13705) for robot imitation learning.

In production this repo is downloaded by the [innate training orchestrator](https://github.com/innate-inc/innate-cloud/tree/main/apps/training-orchestrator), which launches a GPU instance and runs `cloud_run.sh`. See env vars in that script for configuration.

To test locally on a machine with GPUs:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .
PYTHONPATH=act_test:$PYTHONPATH python -m act_test.train_dist --data_dir /path/to/hdf5_data --world_size 1
```
