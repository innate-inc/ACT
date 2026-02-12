# Hyperparameter Sweep Instructions for Innate Policy

This guide explains how to run a hyperparameter sweep for learning rate and weight decay.

## Setup

The sweep configuration is in `sweep_innate.yaml` and will test:
- **Learning rates**: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
- **Weight decays**: [0, 1e-5, 1e-4, 1e-3, 1e-2]
- **Total combinations**: 25 runs (5 LR × 5 WD)

## Step 1: Initialize the Sweep

First, make sure you're logged into W&B:

```bash
wandb login
```

Then initialize the sweep:

```bash
wandb sweep sweep_innate.yaml
```

This will output a sweep ID like:
```
wandb: Created sweep with ID: abc123xyz
wandb: View sweep at: https://wandb.ai/username/innate-policy-sweep/sweeps/abc123xyz
wandb: Run sweep agent with: wandb agent username/innate-policy-sweep/abc123xyz
```

## Step 2: Run Sweep Agent(s)

### Option A: Single Agent (Sequential)
Run one agent that will execute all sweep runs sequentially:

```bash
wandb agent username/innate-policy-sweep/abc123xyz
```

### Option B: Multiple Agents (Parallel)
If you have multiple GPUs or machines, you can run multiple agents in parallel. Each agent will pull the next configuration from the sweep queue:

```bash
# Terminal 1 (uses all available GPUs for one run)
wandb agent username/innate-policy-sweep/abc123xyz

# Terminal 2 (on another machine or after first run completes)
wandb agent username/innate-policy-sweep/abc123xyz
```

**Note**: Each run of `train_innate_dist.py` will use all available GPUs for distributed training. If you want to run multiple experiments in parallel on the same machine, you'll need to manually set `CUDA_VISIBLE_DEVICES`.

### Option C: Run Specific Number of Experiments
Run a specific number of sweep runs:

```bash
wandb agent --count 5 username/innate-policy-sweep/abc123xyz
```

## Step 3: Monitor Results

View your sweep results at:
```
https://wandb.ai/username/innate-policy-sweep/sweeps/abc123xyz
```

The W&B dashboard will show:
- Parallel coordinates plot
- Parameter importance
- Best runs ranked by `val/action_l1_error`

## Modifying the Sweep

### Change Search Method

Edit `sweep_innate.yaml` to change the search strategy:

- **Grid search** (tests all combinations):
  ```yaml
  method: grid
  ```

- **Random search** (samples N random combinations):
  ```yaml
  method: random
  ```

- **Bayesian optimization** (intelligently explores parameter space):
  ```yaml
  method: bayes
  ```

### Add More Parameters

To sweep additional hyperparameters, add them to the `parameters` section:

```yaml
parameters:
  learning_rate:
    values: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
  weight_decay:
    values: [0, 1e-5, 1e-4, 1e-3, 1e-2]
  learning_rate_backbone:
    values: [1e-6, 1e-5, 5e-5]  # Add backbone LR sweep
  num_queries:
    values: [4, 8, 16]  # Add model architecture sweep
```

### Shorter Test Runs

For faster iteration during testing, reduce max_steps:

```yaml
parameters:
  max_steps:
    value: 30000  # Instead of 120000
```

## Tips

1. **Start small**: Test with 2-3 values per parameter first
2. **Monitor early**: Check the first few runs to ensure everything works
3. **Use early stopping**: Add early stopping to the sweep config if some runs are clearly bad:
   ```yaml
   early_terminate:
     type: hyperband
     min_iter: 3
   ```
4. **Resource management**: Each run uses all GPUs - plan accordingly for parallel execution

## Stopping a Sweep

To stop an agent:
- Press `Ctrl+C` in the terminal running the agent

To stop the entire sweep:
- Go to the W&B web interface and click "Stop sweep"
- Or use: `wandb sweep --stop username/innate-policy-sweep/abc123xyz`
