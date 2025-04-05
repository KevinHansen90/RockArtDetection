# RockArtDetection

A research project for a Data Science Master's Degree focusing on detecting Patagonian rock art using advanced object detection techniques and traditional image processing methods. This repository includes datasets, code for pre-processing, model training, and evaluation, as well as detailed documentation.

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
- [Training Workflow](#training-workflow)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Example Usage Flow](#example-usage-flow)
- [License](#license)

## Project Overview

**Goal**: Automatically detect "Animal" figures (and potentially other categories like "Hand") in Patagonian rock art images. Many such images are **very large** (e.g., over 4200×2800 pixels), necessitating techniques like:
- **Tiling**: Breaking large images into smaller, manageable patches.
- **Label Grouping**: Mapping multiple specific classes (e.g., different animal types) into broader categories (e.g., "Animal").
- **Image Preprocessing**: Applying filters (Bilateral, Unsharp Mask, Laplacian, CLAHE) to enhance features potentially beneficial for detection.

This project combines **classical** image processing techniques with **modern** deep learning object detectors (Faster R-CNN, RetinaNet, Deformable DETR) to explore robust detection strategies.

## Repository Structure

A brief description of the main folders relevant to this stage:

```
project/
├── configs/                  # Configuration files (.yaml) for experiments
│   ├── frcnn_config.yaml     # Example config for Faster R-CNN
│   ├── retina_config.yaml    # Example config for RetinaNet
│   └── detr_config.yaml      # Example config for Deformable DETR
│
├── data/
│   ├── raw/                  # Original large images & their YOLO labels (0..N classes)
│   └── tiles/                # Processed dataset tiles
│       ├── base/             # Base tiled dataset (e.g., only 'Animal' class)
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── bilateral/        # Bilateral-filtered dataset splits
│       ├── unsharp/          # Unsharp-filtered dataset splits
│       ├── laplacian/        # Laplacian-filtered dataset splits
│       └── clahe/            # CLAHE-filtered dataset splits
│
├── experiments/              # Output directory for training runs
│   └── {experiment_name}/    # Subfolder for each experiment
│       ├── config.yaml       # Copy of the config used
│       ├── metrics.csv       # Epoch-wise metrics log
│       ├── loss_curve.png    # Plot of training/validation loss
│       ├── map_curve.png     # Plot of validation mAP
│       ├── mar_curve.png     # Plot of validation mAR
│       ├── model_final.pth   # Saved final model checkpoint
│       └── test_visualization.png # Test set evaluation visualization (optional)
│
├── src/
│   ├── preprocessing/
│   │   ├── tile_images.py    # Tiles large images/labels, maps classes
│   │   ├── split_dataset.py  # Splits tiled data into train/val/test
│   │   └── apply_filters.py  # Applies image filters to split datasets
│   ├── datasets/
│   │   └── yolo_dataset.py   # PyTorch Dataset for YOLO-formatted labels
│   ├── models/
│   │   └── detection_models.py # Functions to build detection models
│   ├── training/
│   │   ├── train.py          # Main script to run training experiments
│   │   ├── engine.py         # Core training loop and evaluation logic
│   │   ├── evaluate.py       # Evaluation and visualization on test set
│   │   └── utils.py          # Helper utilities (config loading, plotting, etc.)
│   └── inference/
│       └── inference.py      # Inference script to test fine-tuned models.
│
├── LICENSE                   # Project license file
└── README.md                 # This documentation file
```

**Note**: Additional files/folders (like Docker, notebooks, etc.) can be added as the project grows.

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
    # Key libraries likely include: torch, torchvision, torchmetrics, pyyaml, tqdm, pillow, opencv-python, transformers
    ```

## Data Preprocessing Workflow

Preprocessing is designed to be run locally or once before cloud-based training.

### 1. Tiling

**Script:** `src/preprocessing/tile_images.py`

* Divides large raw images and their corresponding YOLO label files into smaller tiles (e.g., 512x512).
* Uses a class mapping defined within the script (currently focusing on mapping various IDs to "Animal" and "Hand" classes).
* Optionally skips tiles that contain no target objects after mapping (`--skip_empty_tiles`).
* Outputs tiled images and new YOLO label files with mapped class IDs.

**Example Command**:
```bash
python src/preprocessing/tile_images.py \
  --input_images data/raw/images \
  --input_labels data/raw/labels \
  --output_base data/tiles/base \
  --tile_size 512 \
  --allow_partial_tiles \
  --skip_empty_tiles
```

**Output:** data/tiles/base/images/ and data/tiles/base/labels/. Also creates grouped_classes.txt in the output base.

### 2. Splitting

**Script:** `src/preprocessing/split_dataset.py`

Splits the tiled dataset (e.g., data/tiles/base) into train, val, and test subdirectories based on specified ratios.
Uses a random seed for reproducible splits.

**Example Command**:
```bash
python src/preprocessing/split_dataset.py \
  --input_dir data/tiles/base \
  --output_dir data/tiles/base \
  --train_ratio 0.75 \
  --val_ratio 0.05 \
  --test_ratio 0.20 \
  --seed 42
```

**Output:** train/, val/, test/ subfolders within data/tiles/base/.

### 3. Applying Filters

**`apply_filters.py`** (in `src/preprocessing/`)  
- Takes each split (train/val/test), applies one of four filters to the images (Bilateral, Unsharp, Laplacian, CLAHE), and copies the labels unchanged.
- Output directory structure mirrors the base set.
- You run it once per filter to create separate sets for each technique.

**Example Command**:
```bash
# Bilateral
python src/preprocessing/apply_filters.py \
  --base_dir data/tiles/base \
  --output_dir data/tiles/bilateral \
  --filter_type bilateral

# Unsharp
python src/preprocessing/apply_filters.py \
  --base_dir data/tiles/base \
  --output_dir data/tiles/unsharp \
  --filter_type unsharp

# Laplacian
python src/preprocessing/apply_filters.py \
  --base_dir data/tiles/base \
  --output_dir data/tiles/laplacian \
  --filter_type laplacian

# CLAHE
python src/preprocessing/apply_filters.py \
  --base_dir data/tiles/base \
  --output_dir data/tiles/clahe \
  --filter_type clahe
```

**Output:** Separate dataset directories (e.g., data/tiles/bilateral/, data/tiles/unsharp/, etc.), each containing train/, val/, test/ splits.

Thus, you end up with five versions of the dataset (base + 4 filters), each containing train/val/test. This allows you to systematically compare how different filters impact model performance.

## Dataset Details

**Format:** The project uses YOLO format (.txt files) for labels, where each line represents an object: `<class_id> <center_x> <center_y> <width> <height>` (normalized coordinates).

**Loading:** The `src/datasets/yolo_dataset.py` script defines PyTorch Dataset classes (`CustomYOLODataset`, `TestDataset`).

**Model Compatibility:** The dataset loader automatically handles:
- Converting bounding boxes to the format required by the model (absolute `[x1, y1, x2, y2]` for Faster R-CNN/RetinaNet, normalized `[cx, cy, w, h]` for Deformable DETR during training target creation).
- Adjusting class labels (0-indexed for DETR, 1-indexed with background for Faster R-CNN/RetinaNet).
- Providing appropriate collate functions (`collate_fn` or `collate_fn_detr`) for batching.

## Model Zoo

The `src/models/detection_models.py` script provides functions to instantiate various object detection models, adapting them for the number of classes in this project:

- **Faster R-CNN** (fasterrcnn): Based on [`torchvision.models.detection.fasterrcnn_resnet50_fpn_v2`](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html#torchvision.models.detection.fasterrcnn_resnet50_fpn_v2).
- **RetinaNet** (retinanet): Based on [`torchvision.models.detection.retinanet_resnet50_fpn_v2`](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.retinanet_resnet50_fpn_v2.html#torchvision.models.detection.retinanet_resnet50_fpn_v2), including custom head options for focal loss parameterization.
- **Deformable DETR** (deformable_detr): Based on the [Hugging Face implementation of Deformable DETR](https://huggingface.co/SenseTime/deformable-detr), wrapped for compatibility with the training pipeline.

Models are typically loaded with pre-trained weights (e.g., on COCO) and their classification heads are replaced to match the number of target rock art classes.

## Training Workflow

**Configuration:** Training runs are configured using `.yaml` files (examples in `configs/`). These files specify the model type, dataset paths, hyperparameters (learning rate, batch size, epochs, optimizer, scheduler), warmup steps, backbone freezing options, etc.

**Main Script:** `src/training/train.py` is the entry point for starting a training run.

**Engine:** `src/training/engine.py` contains the core logic for the training loop (`train_model`) and validation evaluation (`evaluate_on_dataset`). It handles epoch iteration, forward/backward passes, loss calculation, metric computation (mAP, mAR using torchmetrics), optimizer/scheduler steps, and logging.

**Utilities:** `src/training/utils.py` provides helpers for loading configs, plotting results, saving metrics, and device selection.

**Output:** Each run creates a folder in `experiments/` containing logs, plots, configuration, and the final model checkpoint (`model_final.pth`).

**Example Training Command**:
```bash
python src/training/train.py \
  --config configs/frcnn_config.yaml \
  --experiment frcnn_base_run1
```

## Evaluation

**During Training:** Validation metrics (mAP@0.5, mAR@100) are computed after each epoch using `src/training/engine.py`'s `evaluate_on_dataset` function and logged/plotted.

**After Training:** The `train.py` script can optionally run evaluation on the test set using `src/training/evaluate.py`'s `evaluate_and_visualize` function.

**Visualization:** `evaluate_and_visualize` generates an image (`test_visualization.png` in the experiment folder) showing ground truth boxes (green) and predicted boxes (red, above confidence threshold) side-by-side for samples from the test set. The logic correctly handles coordinate systems and label indexing for all supported models (including the previously discussed fix for DETR ground truth visualization).

## Inference

After training a model, you can use the `src/inference/inference.py` script to run it on new images (or directories of images) and visualize the detections.

**Script:** `src/inference/inference.py`

* Loads a trained model checkpoint (`.pth` file).
* Takes an input image path or directory path.
* Requires the path to the class names file used during training.
* Requires specifying the model type (fasterrcnn, retinanet, deformable_detr).
* Applies the necessary preprocessing for the chosen model.
* Runs inference and draws bounding boxes, class labels, and confidence scores on the images for detections above a specified threshold.
* Saves the output images with drawn predictions to a specified output directory, creating a timestamped subfolder for each run.

**Example Inference Command**:

```bash
python src/inference/inference.py \
    --model-path experiments/your_experiment_name/model_final.pth \
    --input path/to/your/test_images_or_image.jpg \
    --output inference_results/ \
    --classes data/tiles/base/grouped_classes.txt \
    --model-type fasterrcnn \
    --threshold 0.5
```

**Output:** A new subfolder `inference_results/inference_model_final_20250405_145500/` containing the input images with predictions drawn on them.

## Example Usage Flow

**Prepare Data:**
- Place raw images and YOLO labels in `data/raw/`.
- Run `src/preprocessing/tile_images.py` to create `data/tiles/base/`.
- Run `src/preprocessing/split_dataset.py` on `data/tiles/base/`.
- (Optional) Run `src/preprocessing/apply_filters.py` to create filtered datasets (e.g., `data/tiles/bilateral/`).

**Configure Experiment:**
- Copy and modify a template from `configs/` (e.g., `configs/detr_config.yaml`).
- Update dataset paths (e.g., point to `data/tiles/bilateral/`), model parameters, hyperparameters, number of classes (from `grouped_classes.txt`).

**Run Training:**
```bash
python src/training/train.py --config configs/my_detr_bilateral_config.yaml --experiment detr_bilateral_run1
```

**Analyze Results:**

- Check plots (`loss_curve.png`, `map_curve.png`) and logs (`metrics.csv`) in `experiments/detr_bilateral_run1/`.
- Inspect the `test_visualization.png` if test evaluation was enabled.
- The final model is saved as `model_final.pth`.

**Run Inference on New Images:**

```bash
python src/inference/inference.py \
    --model-path experiments/detr_bilateral_run1/model_final.pth \
    --input path/to/some/new_images/ \
    --output inference_output/ \
    --classes data/tiles/base/grouped_classes.txt \
    --model-type deformable_detr \
    --threshold 0.4
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License is a permissive license that is short and to the point. It lets people do anything with your code with proper attribution and without warranty. The full license text is included in the LICENSE file in this repository.
