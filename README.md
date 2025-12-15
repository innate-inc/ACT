# Training ACT policies using Vertex AI

## 1. Upload Your Dataset

First, upload your dataset to Google Cloud Storage. Typically it would be located in `innate-os/primitives/{primitive_name}/data`. It should contain `h5` files of the collected episodes and `dataset_metadata.json`.

```bash
gsutil -m cp -r innate-os/primitives/{primitive_name}/data/* gs://maurice-prod-data/data/{primitive_name}/ 
```

Make sure your account has `storage.objects.delete` rights, so that it can delete temporary files during the upload.

## 2. Authenticate with Google Cloud

If you haven't already, authenticate with Google Cloud to enable the Vertex AI Python SDK:

```bash
gcloud auth application-default login
```

This creates application default credentials that the Vertex AI Python SDK can use.

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

**Note:** The script will check if the checkpoint folder already exists and error if it does, preventing accidental overwrites.

## 4. Monitor Training

After deployment, monitor your training job at:
https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=mauricearm

## 5. Access Your Trained Model

Once training completes, your checkpoints will be available at:
```
gs://maurice-prod-data/ckpts/{primitive_name}/{run_name}/
```

The folder will contain:
- `dataset_stats.pt` - Normalization statistics
- `act_policy_step_*.pth` - Training checkpoints (10 total)
- `act_policy_final.onnx` - Final model in ONNX format for inference