# Training ACT policies using Vertex AI

## 1. Authenticate with Google cloud

You need to obtain the service account key from innate, for the `customer-upload` service account. Then use:

```bash
gcloud auth activate-service-account customer-upload@mauricearm.iam.gserviceaccount.com \
  --key-file=/path/to/your-service-account-key.json
```

Also set
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-service-account-key.json
```

## 2. Upload Your Dataset

First, upload your dataset to Google Cloud Storage. Typically it would be located in `innate-os/primitives/{primitive_name}/data`. It should contain `h5` files of the collected episodes and `dataset_metadata.json`.

```bash
gsutil -m cp -r /home/jetson1/innate-os/primitives/{primitive_name}/data/* gs://maurice-prod-data/data/{primitive_name}/ 
```

## 3. Deploy Training Job to Vertex AI

Run the deployment script with three required arguments:

```bash
./deploy_to_vertex.sh \
    gs://maurice-prod-data/data/{primitive_name} \
    gs://maurice-prod-data/ckpts/{primitive_name} \
    {run_name}
```

**Arguments:**
- `DATA_GCS_PATH`: Path to your training data in GCS
- `OUTPUT_GCS_PATH`: Path where checkpoints will be saved
- `RUN_NAME`: Name for this training run (will be used as the checkpoint folder name)
- `JOB_NAME` (optional): Name for the Vertex AI job

**Example:**

```bash
./deploy_to_vertex.sh \
    gs://maurice-prod-data/data/clean_trash \
    gs://maurice-prod-data/ckpts/clean_trash \
    clean-trash-v1
```

This will create checkpoints in: `gs://maurice-prod-data/ckpts/clean_trash/clean-trash-v1/`

## 4. Download the Checkpoints

When the training is finished, you can download the checkpoints to your local machine:

```bash
gsutil -m cp -r gs://maurice-prod-data/ckpts/{primitive_name}/{run_name}/* /home/jetson1/innate-os/primitives/{primitive_name}/models/
```

**Example:**

```bash
gsutil -m cp -r gs://maurice-prod-data/ckpts/clean_trash/clean-trash-v1/* /home/jetson1/innate-os/primitives/clean_trash/models/
```

The downloaded folder will contain:
- `dataset_stats.pt` - Normalization statistics for input/output data
- `act_policy_step_*.pth` - Training checkpoints (saved every 500 steps)
- `act_policy_final.onnx` - Final model in ONNX format for inference

**Note:** Make sure the destination directory exists before downloading:

```bash
mkdir -p /home/jetson1/innate-os/primitives/{primitive_name}/models/
```