# RockArtDetection

> [!NOTE]
> **Thesis Disclaimer**: The figures, tables, and experimental outputs in the `tesis/` directory were generated using an earlier legacy version of this repository. They are preserved for thesis reference; code and outputs in the current version have been modernized and streamlined.

A research project for detecting Patagonian rock art using advanced deep learning object detectors, traditional image processing filters, and unsupervised motif clustering for stylistic analysis. Supports both local execution (macOS MPS / Linux CUDA) and scalable cloud execution on Google Cloud Platform (Vertex AI, GCS, Artifact Registry, Terraform).

---

## Table of Contents
- [Dataset Availability & Ingestion](#dataset-availability--ingestion)
- [Architecture & Design Overview](#architecture--design-overview)
- [Repository Structure](#repository-structure)
- [Fast Environment Setup](#fast-environment-setup)
- [GCP Infrastructure with Terraform](#gcp-infrastructure-with-terraform)
- [Data Preprocessing Workflow](#data-preprocessing-workflow)
- [Model Zoo](#model-zoo)
- [Running Locally](#running-locally)
- [Running on Google Cloud (Vertex AI)](#running-on-google-cloud-vertex-ai)
- [Model Benchmarking & Evaluation](#model-benchmarking--evaluation)
- [Unit Testing](#unit-testing)
- [License](#license)

---

## Dataset Availability & Ingestion

**Google Drive Dataset**: The original dataset, containing **683 high-resolution images** annotated with **19 classes** (named in Spanish), is publicly available on Google Drive:
👉 **[Download Dataset on Google Drive](https://drive.google.com/drive/u/0/folders/1JU5tohaRw7Rm83S9uUK9KazIPLRebl1x)**

### 1. Direct GCP Cloud Ingestion (No Cloudtop Proxying)
To ingest data directly within GCP internal network infrastructure without streaming files through your local workstation, run the cloud transfer runner:
```bash
python -m src.cloud.transfer_drive_to_gcs \
    --drive_url "https://drive.google.com/drive/u/0/folders/1JU5tohaRw7Rm83S9uUK9KazIPLRebl1x" \
    --gcs_bucket $GCS_BUCKET_NAME \
    --gcs_prefix "data/raw"
```

### 2. Local Download
Alternatively, download locally into your local workspace:
```bash
python -m src.cli download --url "https://drive.google.com/drive/u/0/folders/1JU5tohaRw7Rm83S9uUK9KazIPLRebl1x" --output_dir data/raw
```

---

## Architecture & Design Overview

This project explores object detection fine-tuning and stylistic clustering across a matrix of configurations:
1. **4 Object Detection Architectures**: Faster R-CNN, RetinaNet, Deformable DETR, and Ultralytics YOLO (v5/v8/v11).
2. **5 Image Preprocessing Subsets**: Base (No Filter), Bilateral Filter, Unsharp Mask, Laplacian Edge Boost, CLAHE Contrast Enhancement.
3. **Unsupervised Motif Clustering**: CNN Feature Extractors (ResNet, DenseNet, VGG, InceptionV3) + Clustering Algorithms (K-Means, Agglomerative, Spectral, DBSCAN).

---

## Repository Structure

```
RockArtDetection/
├── pyproject.toml                     # Modern Python package configuration
├── requirements.txt                   # Dependency list
├── Dockerfile                         # Unified Dockerfile for GCP Vertex AI
├── terraform/                         # Terraform HCL for GCP (GCS, Artifact Registry, IAM)
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   └── terraform.tfvars.example
├── configs/                           # Hydra configuration tree
│   ├── defaults.yaml
│   ├── model/                         # fasterrcnn, retinanet, deformable_detr, yolov5
│   ├── data/                          # tiles_base, tiles_bilateral, tiles_unsharp, etc.
│   └── train/                         # cpu_pilot, gpu_t4_pilot, local_gpu, gpu_full, gpu_finetune
├── src/                               # Modular package source
│   ├── cli.py                         # Unified CLI dispatcher (python -m src.cli)
│   ├── pipeline.py                    # End-to-end orchestrator pipeline
│   ├── data/
│   │   ├── download_drive.py          # Google Drive automated downloader
│   │   ├── tile_images.py             # Overlapping tile generator & class remapper
│   │   ├── split_dataset.py           # Train/Val/Test ratio splitter
│   │   ├── apply_filters.py           # Bilateral, Unsharp, Laplacian, CLAHE
│   │   └── yolo_dataset.py            # PyTorch Dataset
│   ├── models/
│   │   └── detection_models.py        # Unified PyTorch Model Zoo (FR-CNN, Retina, DETR, YOLO)
│   ├── training/
│   │   ├── train.py                   # Hydra launcher
│   │   ├── engine.py                  # PyTorch 2.x AMP training & validation loops
│   │   ├── evaluate.py                # Visual collage generator
│   │   └── utils.py                   # Plotting & CSV logging utilities
│   ├── clustering/
│   │   ├── crop_motifs.py             # Bounding-box motif instance cropper
│   │   └── cluster_motifs.py          # Feature extraction & clustering
│   └── cloud/
│       ├── transfer_drive_to_gcs.py   # Direct GCP Drive -> GCS transfer
│       ├── vertex_submit.py           # Vertex AI Custom Job launcher with deduplication & --force
│       ├── vertex_pipeline.py         # Vertex AI Pipeline orchestrator
│       └── gcs_utils.py               # GCS storage utilities
├── results/                           # Experiment outputs and metrics logs
│   ├── models/                        # Trained model checkpoints & metric plots
│   └── summary_full_trainings.csv     # Consolidated metric benchmarks
├── tests/                             # Unit test suite
└── tesis/                             # Legacy Master's thesis LaTeX source
```

---

## Fast Environment Setup

We recommend standard Python 3.10+ virtual environments or [`uv`](https://github.com/astral-sh/uv):

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies in editable mode
pip install -e .
```

---

## GCP Infrastructure with Terraform

Provision required GCP resources (GCS bucket, Artifact Registry Docker repository, and Vertex AI Service Account):

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Set project_id and region in terraform.tfvars

terraform init
terraform apply
```

---

## Data Preprocessing Workflow

### 1. Tile Large Images (with Overlap)
```bash
python -m src.cli tile \
    --input_images data/raw/images \
    --input_labels data/raw/labels \
    --output_base data/tiles/base \
    --tile_size 512 --overlap 100 \
    --allow_partial_tiles --skip_empty_tiles
```

### 2. Split Dataset (80% Train, 5% Val, 15% Test)
```bash
python -m src.cli split \
    --input_dir data/tiles/base \
    --output_dir data/tiles/base \
    --use_ratios --train_ratio 0.80 --val_ratio 0.05 --test_ratio 0.15
```

### 3. Apply Image Preprocessing Filters
```bash
for FILTER in bilateral unsharp laplacian clahe; do
    python -m src.cli filter \
        --base_dir data/tiles/base \
        --output_dir data/tiles/${FILTER} \
        --filter_type ${FILTER}
done
```

---

## Model Zoo

The unified model zoo in `src/models/detection_models.py` supports:
* **RetinaNet**: `retinanet` (ResNet-50 FPN v2 with Focal Loss) - **Top Performing Architecture**
* **Faster R-CNN**: `fasterrcnn` (ResNet-50 FPN v2)
* **Deformable DETR**: `deformable_detr` (`SenseTime/deformable-detr`)
* **Ultralytics YOLO**: `yolov5`

---

## Running Locally

Run training locally (automatically detects `mps` on macOS, `cuda` on Linux, or `cpu`):

```bash
# Run local training
python -m src.cli train model=retinanet data=tiles_base train=local_gpu experiment=retina_local

# Or run the complete unified pipeline locally
python -m src.cli pipeline --filter bilateral --model retinanet --feature-model resnet50 --cluster-algo kmeans
```

---

## Running on Google Cloud (Vertex AI)

### 1. Build and Push Container Image
```bash
export PROJECT_ID=$(gcloud config get-value project)
gcloud builds submit --tag us-central1-docker.pkg.dev/$PROJECT_ID/rockart-docker-repo/rockart-trainer:latest .
```

### 2. Submit Custom Jobs on Vertex AI (with Native GCP Experiments & TensorBoard)
```bash
python -m src.cloud.vertex_submit \
    --project_id $PROJECT_ID \
    --gcs_bucket $PROJECT_ID-rockart-data \
    --container_image us-central1-docker.pkg.dev/$PROJECT_ID/rockart-docker-repo/rockart-trainer:latest \
    --service_account rockart-vertex-sa@$PROJECT_ID.iam.gserviceaccount.com \
    --mode train --model retinanet --filter base --train_config gpu_full \
    --experiment full_retinanet_base \
    --experiment_name rockart-detection \
    --tensorboard projects/$PROJECT_ID/locations/us-central1/tensorboards/$TENSORBOARD_ID \
    --force
```


---

## Model Benchmarking & Evaluation

The training framework evaluates models using standard COCO evaluation metrics (`mAP@50`, `mAR@100`, `F1 Score`), logging visual prediction collages and metric curves to GCS or local directories.


---

## Unit Testing

Run unit tests across model architectures, metrics computation, tiling, filters, and clustering:

```bash
python3 -m unittest discover tests/
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
