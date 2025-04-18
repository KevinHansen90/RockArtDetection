# RockArtDetection

A research project for a Data Science Master's Degree focusing on detecting Patagonian rock art using advanced object detection techniques, traditional image processing methods, and exploring unsupervised clustering for stylistic analysis. This repository includes datasets, code for pre-processing, model training (local and cloud), evaluation, and detailed documentation.

**Dataset:** The dataset, containing **683 images** annotated with **19 classes** (named in Spanish), is available [here](https://drive.google.com/drive/u/0/folders/1JU5tohaRw7Rm83S9uUK9KazIPLRebl1x). The labeling was carried out with the invaluable assistance of Agustina Papu, an archaeologist.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Preprocessing Workflow](#data-preprocessing-workflow)
  - [1. Tiling](#1-tiling)
  - [2. Splitting](#2-splitting)
  - [3. Applying Filters](#3-applying-filters)
- [Dataset Details](#dataset-details)
- [Model Zoo](#model-zoo)
- [Local Training Workflow](#local-training-workflow)
- [Cloud Training Workflow (GCP)](#cloud-training-workflow-gcp)
- [Evaluation](#evaluation)
- [Local Inference](#local-inference)
- [Unsupervised Clustering Workflow](#unsupervised-clustering-workflow)
- [Example Usage Flow](#example-usage-flow)
- [License](#license)

## Project Overview

**Goal**: Automatically detect "Animal" figures (and potentially other categories like "Hand") in Patagonian rock art images and explore unsupervised clustering for stylistic analysis. Many original images are **very large** (e.g., over 4200×2800 pixels), necessitating techniques like:
- **Tiling with Overlap**: Breaking large images into smaller, overlapping patches to ensure object capture.
- **Label Grouping**: Mapping multiple specific classes into broader categories (e.g., "Animal").
- **Image Preprocessing Filters**: Applying techniques (Bilateral, Unsharp Mask, Laplacian, CLAHE) to enhance features.

This project combines **classical** image processing with **modern** deep learning object detectors (Faster R-CNN, RetinaNet, Deformable DETR, YOLOv5) and unsupervised learning (CNN feature extraction + clustering) to explore robust analysis strategies. Experiments are scaled using **Google Cloud Platform (GCP)**.

## Repository Structure

```
project/
├── configs/                  # Configuration files (.yaml) for experiments
│   ├── gcp/                  # Configs for GCP runs (pointing to GCS paths)
│   │   └── base_detr.yaml    # Example GCP config
│   ├── yolov5/               # YOLOv5 data config files
│   │   └── base_data.yaml    # Example YOLOv5 data config for GCS
│   ├── frcnn_config.yaml     # Example local config for Faster R-CNN
│   ├── retina_config.yaml    # Example local config for RetinaNet
│   └── detr_config.yaml      # Example local config for Deformable DETR
│
├── data/
│   ├── raw/                  # Original large images & their YOLO labels
│   ├── clustering/           # Data generated for clustering steps
│   │   └── cropped/          # Example output of motif cropping
│   └── tiles/                # Processed dataset tiles (e.g., with overlap)
│       ├── base/             # Base tiled dataset with overlap
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── bilateral/        # Bilateral-filtered dataset splits
│       ├── unsharp/          # Unsharp-filtered dataset splits
│       # ... (etc. for other filters) ...
│
├── experiments/              # Output directory for LOCAL training runs
│   └── {experiment_name}/    # Subfolder for each local experiment
│       ├── ... (logs, plots, model) ...
│
├── src/
│   ├── clustering/           # Scripts for clustering analysis
│   │   └── cluster_motifs.py # Feature extraction and clustering script
│   │   └── crop_motifs.py    # Extracts/resizes specific motifs
│   ├── preprocessing/
│   │   ├── tile_images.py    # Tiles large images (with overlap) & labels
│   │   ├── split_dataset.py  # Splits tiled data into train/val/test
│   │   ├── apply_filters.py  # Applies image filters to split datasets
│   ├── datasets/
│   │   └── yolo_dataset.py   # PyTorch Dataset for custom models
│   ├── models/
│   │   └── detection_models.py # Functions for Faster R-CNN, RetinaNet, DETR
│   ├── training/
│   │   ├── train.py          # Main script for local training runs
│   │   ├── engine.py         # Core training loop and evaluation logic
│   │   ├── evaluate.py       # Evaluation and visualization
│   │   └── utils.py          # Helper utilities
│   └── inference/
│       └── inference.py      # Local inference script
│
├── Dockerfile                # Dockerfile for custom training environment
├── Dockerfile.yolov5         # Dockerfile for YOLOv5 training environment
├── requirements.txt          # Python package dependencies
├── LICENSE                   # Project license file
└── README.md                 # This documentation file
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/KevinHansen90/RockArtDetection
    cd RockArtDetection
    ```

2.  **Create a Python environment:** (e.g., using conda or venv)
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install System Dependencies:**
    * **Docker:** Required for containerizing the training environments for GCP. Follow the official Docker installation guide for your OS.
    * **Google Cloud SDK (`gcloud`):** Required for interacting with GCP (uploading data, submitting jobs). Follow the official GCP SDK installation guide.

## Data Preprocessing Workflow

Preprocessing is typically run locally once before uploading datasets to the cloud for training.

### 1. Tiling

**Script:** `src/preprocessing/tile_images.py`

* Divides large raw images and their corresponding YOLO label files into smaller patches (e.g., 512x512).
* Includes an option to generate **overlapping tiles** (`--overlap` argument) to better handle objects that span tile boundaries. This is the recommended approach.
* Uses a class mapping defined within the script to map original label IDs to broader categories (e.g., "Animal", "Hand") specified in `labels_of_interest`.
* Calculates correct relative YOLO coordinates for all object parts visible within each (potentially overlapping) tile.
* Optionally skips saving tiles that contain no target objects after mapping (`--skip_empty_tiles`).
* Optionally allows partial tiles along the right and bottom edges of the original image (`--allow_partial_tiles`).
* Outputs tiled images and new YOLO label files. Filenames include coordinates when overlap is used (e.g., `imagename_x512_y1024_s512.jpg`).
* Writes a `grouped_classes.txt` file listing the final class IDs and names used in the tiled dataset.

**Example Command (with 100px overlap):**
```bash
python src/preprocessing/tile_images.py \
  --input_images data/raw/images \
  --input_labels data/raw/labels \
  --output_base data/tiles/base \
  --tile_size 512 \
  --overlap 100 \
  --allow_partial_tiles \
  --skip_empty_tiles \
  --image_ext .jpg
```

**Output:** A new dataset directory (e.g., `data/tiles/base/`) containing `images/` and `labels/` subfolders with the overlapping tiles and their corresponding annotations, plus the `grouped_classes.txt` file in the parent directory (e.g. `data/tiles/`).

### 2. Splitting

**Script:** `src/preprocessing/split_dataset.py`

* Splits a tiled dataset (e.g., `data/tiles/base`) into `train/`, `val/`, and `test/` subdirectories.
* Can split based on specified ratios (`--use_ratios` and `--train_ratio`, etc.) or specific numbers of images (`--use_numbers` and `--train_num`, etc.).
* Uses a random seed (`--seed`) for reproducible splits.

**Example Command (Ratio-based on overlapped data):**
```bash
python src/preprocessing/split_dataset.py \
  --input_dir data/tiles/base \
  --output_dir data/tiles/base \
  --use_ratios \
  --train_ratio 0.80 \
  --val_ratio 0.05 \
  --test_ratio 0.15 \
  --seed 42 \
  --image_ext .jpg
```

**Output:** `train/`, `val/`, `test/` subfolders created within the specified input/output directory (e.g., `data/tiles/base/`).

### 3. Applying Filters

**Script:** `src/preprocessing/apply_filters.py`

* Takes a split dataset (e.g., `data/tiles/base`) as input.
* Applies one of four image filters (Bilateral, Unsharp, Laplacian, CLAHE) to the images in the `train/`, `val/`, and `test/` splits.
* Copies the corresponding label files unchanged.
* Saves the results to a new output directory (e.g., `data/tiles/bilateral`).
* Run this script multiple times (once per filter type) to generate different filtered dataset versions for experimentation.

**Example Command (Bilateral filter on overlapped data):**
```bash
python src/preprocessing/apply_filters.py \
  --base_dir data/tiles/base \
  --output_dir data/tiles/bilateral \
  --filter_type bilateral
```

**Output:** Separate dataset directories (e.g., `data/tiles/bilateral/`, `data/tiles/unsharp/`, etc.), each containing train/val/test splits with filtered images.

## Dataset Details

**Label Format:** The project primarily uses YOLO format (.txt files) for labels: `<class_id> <center_x> <center_y> <width> <height>` (normalized coordinates relative to the image/tile dimensions).

**Dataset Structure:** Tiled datasets are expected in a standard structure with train/val/test splits containing images/ and labels/ subdirectories. 

```
data/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**YOLOv5 Configuration:** For training with YOLOv5, an additional `data.yaml` file is required for each dataset version, specifying paths (can be GCS paths) and class info. See the example structure below (paths should point to GCS locations for cloud training):
```
# Example data.yaml for YOLOv5 (paths should be GCS paths for cloud training)
# path: gs://your-bucket-name/data/tiles/base_overlap100 # Optional root
train: gs://your-bucket-name/data/tiles/base_overlap100/train/images
val: gs://your-bucket-name/data/tiles/base_overlap100/val/images
test: gs://your-bucket-name/data/tiles/base_overlap100/test/images # Optional

nc: 2  # Number of classes
names: ['Animal', 'Hand'] # Class names
```

**Custom Model Loading:** The `src/datasets/yolo_dataset.py` script defines PyTorch `Dataset` classes (`CustomYOLODataset`, `TestDataset`) used by `src/training/train.py` for Faster R-CNN, RetinaNet, and Deformable DETR. It handles coordinate/label format adjustments automatically based on the model type.

## Model Zoo

This project explores several object detection models:

* **Faster R-CNN** (`fasterrcnn`): Implemented via `src/models/detection_models.py` using [`torchvision.models.detection.fasterrcnn_resnet50_fpn_v2`](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html#torchvision.models.detection.fasterrcnn_resnet50_fpn_v2).
* **RetinaNet** (`retinanet`): Implemented via `src/models/detection_models.py` using [`torchvision.models.detection.retinanet_resnet50_fpn_v2`](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.retinanet_resnet50_fpn_v2.html#torchvision.models.detection.retinanet_resnet50_fpn_v2).
* **Deformable DETR** (`deformable_detr`): Implemented via `src/models/detection_models.py` using the [Hugging Face `transformers` library (`SenseTime/deformable-detr`)](https://huggingface.co/SenseTime/deformable-detr).
* **YOLOv5** (`yolov5s`, `yolov5m`, etc.): Trained using the [Ultralytics YOLOv5 repository](https://github.com/ultralytics/yolov5) scripts directly, not `src/training/train.py`. Requires the specific `data.yaml` format.

Models are fine-tuned starting from COCO pre-trained weights.

## Local Training Workflow

This section describes running training experiments locally using `src/training/train.py`. This is suitable for debugging, smaller tests, or if cloud resources are unavailable, but likely slow for full dataset training. This workflow applies only to Faster R-CNN, RetinaNet, and Deformable DETR defined in this repository.

**Configuration:** Use `.yaml` files in `configs/` (e.g., `configs/detr_config.yaml`). These specify model type, **local** dataset paths, hyperparameters, etc.

**Main Script:** `src/training/train.py`

**Engine & Utilities:** `src/training/engine.py`, `src/training/utils.py` provide the core logic.

**Output:** Runs are saved locally in `experiments/{experiment_name}/`, containing logs, plots, config copy, and the final model (`model_final.pth`).

**Example Local Training Command:**
```bash
python src/training/train.py \
  --config configs/detr_config.yaml \
  --experiment detr_local_run1
```

## Cloud Training Workflow (GCP)

For efficient and scalable training of multiple model/dataset combinations, Google Cloud Platform (GCP) using Vertex AI is recommended.

**Purpose:** To run the 20 planned experiments (4 models x 5 dataset versions) cost-effectively within the target budget (~$300), leveraging cloud GPUs.

**Overview:**
1.  **Local Preprocessing:** Prepare all 5 dataset versions (e.g., `base_overlap100`, `bilateral_overlap100`, etc.) locally, including tiling with overlap, splitting, and applying filters.
2.  **GCS Upload:** Upload the prepared datasets and necessary config files (`.yaml` for custom models, `data.yaml` for YOLOv5) to a Google Cloud Storage (GCS) bucket.
3.  **Docker Containers:** Create two Docker images (using `Dockerfile` and `Dockerfile.yolov5`) containing the code and dependencies for:
    * Your custom training script (`src/training/train.py`).
    * The Ultralytics YOLOv5 training script.
    Push these images to Google Container Registry (GCR) or Artifact Registry.
4.  **Vertex AI Custom Jobs:** Submit training jobs (one per experiment) to Vertex AI Training.
    * Use the appropriate Docker container for each job.
    * Configure jobs to use cost-effective **preemptible/spot NVIDIA T4 GPUs**.
    * Pass GCS paths for data, configs, and output directories as arguments.
    * Ensure training scripts save checkpoints frequently to GCS to handle potential preemptions.
5.  **Monitoring:** Track job progress and costs in the GCP console. Set budget alerts.
6.  **GCS Results:** Completed jobs will save outputs (checkpoints, logs, plots) to the specified GCS bucket.
7.  **Local Analysis:** Download the final trained models (`.pth` or `.pt` files) from GCS for local inference, evaluation, and clustering analysis.

**Cost Optimization:** The primary cost-saving techniques are using **preemptible/spot T4 GPUs** and performing all data preprocessing locally.

**Preliminary Runs:** It is highly recommended to first run each job configuration on GCP for only 1-3 epochs to verify the entire pipeline (container works, data access okay, training starts, outputs saved) before launching full training runs.

**Example Job Submission (Conceptual - requires `gcloud` CLI):**
*For Custom Model (`train.py`):*
```bash
# Example - Replace placeholders with your actual values!
gcloud ai custom-jobs create \
  --project=your-project-id \
  --region=us-central1 \
  --display-name="train_DETR_bilateral_overlap" \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri="gcr.io/your-project-id/rockart-trainer:latest" \
  --args="--config=gs://your-bucket-name/configs/gcp/bilateral_detr.yaml,--experiment=DETR_bilateral_run1_$(date +%Y%m%d_%H%M%S)" \
  --base-output-directory="gs://your-bucket-name/experiments/" \
  --enable-scheduling-termination
```

*For YOLOv5 (`yolov5/train.py`):*
```bash
# Example - Replace placeholders with your actual values!
gcloud ai custom-jobs create \
  --project=your-project-id \
  --region=us-central1 \
  --display-name="train_YOLOv5m_bilateral_overlap" \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri="gcr.io/your-project-id/yolov5-trainer:latest" \
  --args="--img=640,--batch=16,--epochs=50,--data=gs://your-bucket-name/configs/yolov5/bilateral_overlap100_data.yaml,--weights=yolov5m.pt,--project=gs://your-bucket-name/experiments/YOLOv5m_bilateral_run1/,--name='',--cache=ram" \
  --base-output-directory="gs://your-bucket-name/experiments/YOLOv5m_bilateral_run1/" \
  --enable-scheduling-termination
```

## Evaluation

**During Training:** Validation metrics (mAP@0.5, mAR@100, defined in `src/training/engine.py`) are computed periodically (e.g., every epoch) and logged to CSV and plotted when training via `src/training/train.py`. YOLOv5 training also logs validation metrics. Monitoring these metrics is crucial for deciding when to stop training (early stopping) or adjust hyperparameters.

**After Training (Local):**
* **Custom Models:** `src/training/evaluate.py` (`evaluate_and_visualize` function) can generate a side-by-side visualization of GT vs Predictions on a test set using a downloaded model checkpoint.
* **YOLOv5:** Use the Ultralytics `val.py` script or load the model via `torch.hub` to evaluate on a test set.

## Local Inference

After downloading trained model checkpoints (`.pth` or `.pt`) from GCS (or using models trained locally), you can perform inference on new images locally.

**Script (Custom Models):** `src/inference/inference.py`
* Loads a `.pth` checkpoint for Faster R-CNN, RetinaNet, or Deformable DETR.
* Takes input image(s)/directory, class file, model type, and threshold.
* Outputs images with predictions drawn.
* Can generate side-by-side GT vs Pred comparisons if ground truth labels are provided.

**Example Command (Custom Model):**
```bash
python src/inference/inference.py \
    --model-path path/to/downloaded/model_final.pth \
    --input path/to/your/test_images/ \
    --labels path/to/your/test_labels/ `# Optional: For comparison output` \
    --output inference_comparisons/ \
    --classes data/grouped_classes.txt `# Or path to GCS downloaded one` \
    --model-type deformable_detr `# Match the trained model` \
    --threshold 0.4
```

**YOLOv5 Inference:** Use the Ultralytics YOLOv5 repository's tools or `torch.hub`.
```bash
# First, clone yolov5 repo if not done: git clone https://github.com/ultralytics/yolov5
# Then run detection:
python yolov5/detect.py \
    --weights path/to/downloaded/yolov5_best.pt \
    --source path/to/your/test_images/ \
    --conf-thres 0.4 \
    --imgsz 640 \
    --project inference_results/ \
    --name yolov5_preds
```

## Unsupervised Clustering Workflow

This workflow uses the generated object detection datasets to explore stylistic patterns in motifs (e.g., "Animal").

### 1. Motif Extraction

**Script:** `src/clustering/crop_motifs.py`

* Extracts specific motifs (based on class ID) from a tiled dataset (e.g., `data/tiles/base/train/`) using bounding box labels.
* Optionally resizes cropped motifs (`--resize-dim`) for consistent input to feature extractors.
* Saves each cropped motif as a separate image file.

**Example Command (Extracting 'Animal' ID 0, resizing):**
```bash
python src/preprocessing/crop_motifs.py \
    --images data/tiles/base/train/images \
    --labels data/tiles/base/train/labels \
    --output data/clustering/cropped \
    --class-id 0 \
    --image-ext .jpg \
    --resize-dim 224
```

**Output:** A directory containing individual image files for each extracted motif (e.g., `data/clustering/cropped/`).

### 2. Feature Extraction & Clustering

**Script:** `src/clustering/cluster_motifs.py`

* **Input:** Directory of cropped motif images.
* **Feature Extraction:** Uses a selected pre-trained CNN (`--feature-model`: ResNet, VGG, DenseNet, InceptionV3) to extract a feature vector for each motif.
* **Clustering:** Applies a selected algorithm (`--cluster-algo`: K-Means, Agglomerative, Spectral, DBSCAN) to group motifs based on feature similarity. Requires cluster count (`--num-clusters`) or DBSCAN parameters (`--dbscan_eps`, `--dbscan_min_samples`).
* **Output:** Saves results (`--output`), including a CSV mapping files to clusters, optionally copies images to cluster folders (`--copy-images`), and optionally generates a t-SNE plot (`--visualize-tsne`).
* **Evaluation:** Assess cluster quality using internal metrics (e.g., Silhouette Score) or external metrics (NMI, ARI) if reference labels exist, alongside qualitative expert review.

**Example Command (ResNet50 features, K-Means, K=5):**
```bash
python src/clustering/cluster_motifs.py \
    --input data/clustering/cropped \
    --output clustering_results/animals_resnet50_kmeans_k5 \
    --feature-model resnet50 \
    --cluster-algo kmeans \
    --num-clusters 5 \
    --copy-images \
    --visualize-tsne
```

## Example Usage Flow

This outlines the end-to-end process using the recommended cloud workflow:

1.  **Local Preprocessing:**
    * Tile raw data with overlap using `src/preprocessing/tile_images.py`.
    * Split the resulting tiled data using `src/preprocessing/split_dataset.py`.
    * Apply desired image filters (e.g., bilateral) to the split dataset using `src/preprocessing/apply_filters.py` for each filter type.

2.  **GCP Setup & Upload:**
    * Create a Google Cloud Storage (GCS) bucket.
    * Upload the final processed dataset versions (base + filtered splits) to the GCS bucket.
    * Create and upload the necessary configuration files (`.yaml` for custom models, `data.yaml` for YOLOv5, ensuring they point to GCS paths) to GCS.

3.  **Containerization:**
    * Build the Docker images defined in `Dockerfile` (for custom models) and `Dockerfile.yolov5`.
    * Push the built container images to Google Container Registry (GCR) or Artifact Registry associated with your GCP project.

4.  **Cloud Training (Vertex AI):**
    * Submit preliminary (1-3 epoch) Vertex AI Custom Training jobs for all planned experiments (e.g., 4 models x 5 datasets = 20 jobs) using the appropriate container image and configuration files on GCS. Use preemptible/spot T4 GPUs. Verify the pipeline integrity (data loading, training start, output saving).
    * Monitor these short test runs and debug any configuration or access issues.
    * Submit the full training jobs with the target number of epochs, again using preemptible/spot T4 GPUs.
    * Monitor the full training runs and project costs via the GCP console.

5.  **Download Results:**
    * Once training jobs are complete, download the final model checkpoints (`.pth` or `.pt` files) and any relevant logs or metrics from the experiment output directories in GCS using `gsutil`.

6.  **Local Inference:**
    * Perform inference on new images locally using the downloaded model checkpoints with `src/inference/inference.py` (for custom models) or the Ultralytics YOLOv5 tools (`detect.py` or `torch.hub`).

7.  **Clustering Analysis (Optional):**
    * Extract specific motifs (e.g., animals) from a dataset split using `src/clustering/crop_motifs.py`, optionally resizing them.
    * Run feature extraction and clustering on the cropped motifs using `src/clustering/cluster_motifs.py` with desired CNN feature extractors and clustering algorithms. Analyze the output CSV and optional visualizations.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

The MIT License is a permissive license that is short and to the point. It lets people do anything with your code with proper attribution and without warranty. The full license text is included in the LICENSE file in this repository.
