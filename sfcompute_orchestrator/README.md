# Lambda Labs Training Orchestrator

Event-driven architecture for managing GPU training jobs on Lambda Labs. This service runs on GCP Cloud Run and automatically:

1. **Monitors availability** - Continuously polls Lambda Labs for GPU instance availability and pricing
2. **Queues jobs** - Accepts training job requests via HTTP API
3. **Requests approval** - Sends Discord notifications for manual approval before launching
4. **Executes training** - Launches instances, runs training, uploads results to GCS
5. **Sends callbacks** - Notifies your service when jobs complete/fail

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Price Monitor  │────▶│  Redis Cache    │◀────│  Job Executor   │
│ (Lambda Labs)   │     │  (prices/queue) │     │ (launch/train)  │
└─────────────────┘     └────────▲────────┘     └─────────────────┘
                                 │
                        ┌────────┴────────┐
                        │    Job API      │
                        │  (HTTP/REST)    │
                        └─────────────────┘
                                 ▲
                                 │
                        ┌────────┴────────┐
                        │   Your App      │
                        │   POST /jobs    │
                        └─────────────────┘
```

## Quick Start

### Prerequisites

1. Lambda Labs API key (from https://cloud.lambdalabs.com/api-keys)
2. GCP project with APIs enabled
3. Discord webhook (optional, for approval workflow)

### Deploy to GCP

```bash
# 1. Enable required APIs
gcloud services enable --project=YOUR_PROJECT \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  containerregistry.googleapis.com \
  redis.googleapis.com \
  vpcaccess.googleapis.com \
  compute.googleapis.com \
  servicenetworking.googleapis.com

# 2. Copy and configure .env
cp env.template .env
# Edit .env with your settings (especially LAMBDA_API_KEY)

# 3. Deploy
./deploy_to_gcp.sh
```

### Submit a Job

```bash
# Get auth token
TOKEN=$(gcloud auth print-identity-token)

# Submit training job
curl -X POST https://YOUR-SERVICE.run.app/jobs \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "data_gcs_path": "gs://your-bucket/training-data",
    "output_gcs_path": "gs://your-bucket/checkpoints",
    "max_steps": 120000,
    "batch_size": 96
  }'

# Check job status
curl -H "Authorization: Bearer $TOKEN" https://YOUR-SERVICE.run.app/jobs

# View prices
curl -H "Authorization: Bearer $TOKEN" https://YOUR-SERVICE.run.app/prices
```

## API Reference

### Submit Training Job

```http
POST /jobs
Authorization: Bearer <token>
Content-Type: application/json

{
  "data_gcs_path": "gs://bucket/path/to/data",   // Required
  "output_gcs_path": "gs://bucket/outputs",       // Optional
  "callback_url": "https://your-service/done",    // Optional - POST on completion/failure
  "batch_size": 96,                               // Optional (default: 96)
  "max_steps": 120000,                            // Optional (default: 120000)
  "chunk_size": 30,                               // Optional (default: 30)
  "learning_rate": "5e-5",                        // Optional
  "num_workers": 4,                               // Optional
  "min_gpus": 1,                                  // Optional (default: 1)
  "max_gpus": 8,                                  // Optional (default: 8)
  "max_duration_hours": 24,                       // Optional - for cost estimation
  "max_total_cost": 500.0,                        // Optional - total budget cap
  "max_spend": 5.00                               // Optional - max $/GPU/hr willing to pay
}
```

**Response:**
```json
{
  "job_id": "abc123",
  "status": "pending",
  "message": "Job submitted successfully",
  "queue_position": 1
}
```

### Callback Payload

When a job completes or fails, if `callback_url` was provided, a POST is sent:

```json
{
  "job_id": "abc123",
  "status": "completed",
  "data_gcs_path": "gs://...",
  "output_gcs_path": "gs://...",
  "instance_type": "gpu_8x_a100",
  "region": "us-west-1",
  "created_at": "2025-01-01T00:00:00",
  "started_at": "2025-01-01T00:05:00",
  "completed_at": "2025-01-01T02:00:00",
  "error_message": null,
  "buy_option": {...}
}
```

### Other Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/jobs` | GET | List all jobs |
| `/jobs/{job_id}` | GET | Get job status |
| `/jobs/{job_id}/cancel` | POST | Cancel a job |
| `/jobs/{job_id}/logs` | GET | Fetch training logs via SSH |
| `/prices` | GET | View cached instance prices |
| `/health` | GET | Health check (no auth required) |
| `/admin/clear-all` | POST | Clear queue and cache |

## Instance Type Priority

The orchestrator prioritizes instance types in this order:

1. **8x H100 SXM5** (best performance)
2. **8x H100 SXM5-GDR**
3. **8x H100 PCIe**
4. **8x A100 80GB SXM4**
5. **8x A100 40GB**
6. **4x H100/A100 variants**
7. **2x variants**
8. **1x variants**

## Configuration

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `LAMBDA_API_KEY` | Lambda Labs API key |
| `GCP_PROJECT` | GCP project ID |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `localhost` | Redis server host |
| `REDIS_PORT` | `6379` | Redis server port |
| `DISCORD_WEBHOOK_URL` | `` | Discord webhook for notifications |
| `DISCORD_REQUIRE_APPROVAL` | `true` | Require Discord approval before launching |
| `DISCORD_APPROVAL_TIMEOUT` | `300` | Seconds to wait for approval |
| `GITHUB_TOKEN` | `` | GitHub PAT for private repo cloning |
| `GCS_SERVICE_ACCOUNT_KEY_B64` | `` | Base64-encoded GCS service account key |
| `DRY_RUN` | `false` | Skip actual instance launches |

## Discord Approval Workflow

When enabled, all instance launches require manual Discord approval:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Job Ready   │───▶│ Send to     │───▶│ User Clicks │───▶│ Execute or  │
│ to Execute  │    │ Discord     │    │ Approve/    │    │ Requeue     │
│             │    │             │    │ Reject      │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Discord Setup

1. Create a webhook in your Discord channel (Settings → Integrations → Webhooks)
2. Add `DISCORD_WEBHOOK_URL` to your `.env` file
3. Deploy - the service will send approval requests to Discord

### Discord Message Example

```
🚀 Lambda Labs Instance Launch - Approval Required

📋 Job ID: abc123
📁 Data Path: gs://bucket/training-data
🖥️ Instance: gpu_8x_a100 (8 GPUs)
🌍 Region: us-west-1
💰 Price: $10.32/hr × 24h = $247.68 total

[✅ APPROVE] | [❌ REJECT]

Expires in 5 minutes
```

## Job Status Flow

```
pending → selecting → awaiting_approval → buying → provisioning → running → completed
                            │                │                        │
                            │                │                        └──▶ failed
                            │                │
                            │                └──▶ [launch failed] → pending (retry)
                            │
                            └──▶ [rejected/expired] → pending (back of queue)
```

## Training Execution

The orchestrator uses Lambda Labs' `user_data` (cloud-init) to bootstrap instances:

1. **Instance Launch** - Creates Lambda Labs instance with startup script
2. **Auto-Setup** - Script installs dependencies, clones repo, downloads data from GCS
3. **Training** - Runs distributed training across all GPUs
4. **Completion Detection** - Monitors for `/tmp/training_complete` marker via SSH
5. **Cleanup** - Uploads checkpoints to GCS, terminates instance

### Startup Script Features

- Automatic Python venv setup
- PyTorch installation (with B200/sm_100 nightly support)
- GCS data download via `gcloud storage cp`
- Private GitHub repo cloning (with GITHUB_TOKEN)
- Distributed training with `torchrun`
- Checkpoint upload to GCS on completion

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# Run locally (dry-run mode)
python -m sfcompute_orchestrator --dry-run

# Run with debug logging
python -m sfcompute_orchestrator --debug
```

## Troubleshooting

### SSH Not Configured

The orchestrator needs an SSH key to monitor training:

```bash
# Generate SSH key via API
TOKEN=$(gcloud auth print-identity-token)
curl -X POST https://YOUR-SERVICE.run.app/debug/ssh-keys/generate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "orchestrator-key"}'
```

### Job Stuck in Provisioning

Lambda instances can take 2-5 minutes to boot. If stuck longer:

1. Check Lambda Labs dashboard for instance status
2. SSH into instance to check `/var/log/training-startup.log`
3. Check Cloud Run logs for executor errors

### Training Failed

Check logs via API:
```bash
curl -H "Authorization: Bearer $TOKEN" \
  https://YOUR-SERVICE.run.app/jobs/{job_id}/logs
```

Or SSH directly:
```bash
ssh -i ~/.ssh/lambda_key ubuntu@INSTANCE_IP
tail -f /var/log/training-startup.log
```

## Security

- **Cloud Run Auth**: All API endpoints require GCP identity token or API key
- **Discord Callbacks**: Protected by secret token (auto-generated)
- **SSH Keys**: Stored encrypted in Redis, used for instance monitoring
- **GCS Credentials**: Passed via base64-encoded environment variable

## License

MIT
