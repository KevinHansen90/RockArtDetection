#!/usr/bin/env python3
"""
Unified End-to-End Orchestrator Pipeline for RockArtDetection.
Runs data downloading, tiling, splitting, filtering, training, and motif clustering.
"""

import os
import sys
import argparse
from src.data.download_drive import download_from_gdrive
from src.data.tile_images import tile_image_and_labels_with_overlap, write_grouped_classes
from src.data.split_dataset import copy_image_and_label
from src.data.apply_filters import process_subset, apply_bilateral, apply_unsharp, apply_laplacian, apply_clahe
from src.training.train import run_training
from src.clustering.crop_motifs import crop_resize_and_save_motifs
from src.clustering.cluster_motifs import run_single


def run_full_pipeline(
    gdrive_url: str = "https://drive.google.com/drive/u/0/folders/1JU5tohaRw7Rm83S9uUK9KazIPLRebl1x",
    raw_dir: str = "data/raw",
    tiles_base_dir: str = "data/tiles/base",
    tile_size: int = 512,
    overlap: int = 100,
    filter_type: str = "bilateral",
    model_name: str = "retinanet",
    feature_model: str = "resnet50",
    cluster_algo: str = "kmeans",
    num_clusters: int = 5,
    skip_download: bool = False,
):
    print("==================================================")
    print("   ROCKARTDETECTION UNIFIED PIPELINE LAUNCHER     ")
    print("==================================================")

    # 1. Download dataset
    if not skip_download:
        print("\n--- STEP 1: Downloading Dataset from Google Drive ---")
        download_from_gdrive(gdrive_url, raw_dir, is_folder=True)
    else:
        print("\n--- STEP 1: Skipping Download (using existing raw data) ---")

    # 2. Tiling
    print("\n--- STEP 2: Tiling Large Images ---")
    raw_imgs = os.path.join(raw_dir, "images")
    raw_lbls = os.path.join(raw_dir, "labels")
    out_img_dir = os.path.join(tiles_base_dir, "images")
    out_lbl_dir = os.path.join(tiles_base_dir, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    if os.path.exists(raw_imgs):
        for img_file in os.listdir(raw_imgs):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                base_name = os.path.splitext(img_file)[0]
                img_path = os.path.join(raw_imgs, img_file)
                lbl_path = os.path.join(raw_lbls, base_name + ".txt")
                tile_image_and_labels_with_overlap(
                    img_path, lbl_path, out_img_dir, out_lbl_dir,
                    tile_size, overlap, allow_partial_tiles=True, skip_empty_tiles=True
                )
        write_grouped_classes(tiles_base_dir)

    # 3. Filtering
    print(f"\n--- STEP 3: Applying Preprocessing Filter ({filter_type}) ---")
    filtered_dir = f"data/tiles/{filter_type}"
    filter_fns = {
        "bilateral": apply_bilateral,
        "unsharp": apply_unsharp,
        "laplacian": apply_laplacian,
        "clahe": apply_clahe,
    }
    if filter_type in filter_fns:
        for subset in ["train", "val", "test"]:
            if os.path.exists(os.path.join(tiles_base_dir, subset)):
                process_subset(tiles_base_dir, filtered_dir, subset, filter_fns[filter_type])

    # 4. Training
    print(f"\n--- STEP 4: Training Detection Model ({model_name}) ---")
    train_cfg = {
        "model_type": model_name,
        "data_root": filtered_dir if os.path.exists(filtered_dir) else tiles_base_dir,
        "classes_file": "data/grouped_classes.txt",
        "experiment": f"pipeline_run_{model_name}_{filter_type}",
        "num_epochs": 2,
        "batch_size": 2,
        "num_workers": 2,
    }
    run_training(train_cfg)

    # 5. Motif Crop & Clustering
    print(f"\n--- STEP 5: Motif Extraction & Clustering ({feature_model} + {cluster_algo}) ---")
    cropped_dir = "data/clustering/motifs"
    os.makedirs(cropped_dir, exist_ok=True)
    train_imgs_path = os.path.join(tiles_base_dir, "train", "images")
    train_lbls_path = os.path.join(tiles_base_dir, "train", "labels")

    if os.path.exists(train_imgs_path):
        for f in os.listdir(train_imgs_path):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                ip = os.path.join(train_imgs_path, f)
                lp = os.path.join(train_lbls_path, os.path.splitext(f)[0] + ".txt")
                crop_resize_and_save_motifs(ip, lp, cropped_dir, target_class_id=0, resize_dim=224)

    if os.path.exists(cropped_dir) and len(os.listdir(cropped_dir)) > 0:
        out_cluster_dir = f"clustering_results/{feature_model}_{cluster_algo}"
        run_single(
            input_dir=cropped_dir,
            output_dir=out_cluster_dir,
            model_name=feature_model,
            algo=cluster_algo,
            k=num_clusters,
            make_plots=True,
        )

    print("\n==================================================")
    print("   UNIFIED PIPELINE EXECUTED SUCCESSFULLY!        ")
    print("==================================================")


def main():
    parser = argparse.ArgumentParser(description="Unified Pipeline Orchestrator for RockArtDetection.")
    parser.add_argument("--filter", default="bilateral", help="Filter type to apply")
    parser.add_argument("--model", default="retinanet", help="Detection model architecture")
    parser.add_argument("--feature-model", default="resnet50", help="Clustering feature extractor")
    parser.add_argument("--cluster-algo", default="kmeans", help="Clustering algorithm")
    parser.add_argument("--num-clusters", type=int, default=5, help="Number of clusters (k)")
    parser.add_argument("--skip-download", action="store_true", help="Skip Google Drive downloading step")
    args = parser.parse_args()

    run_full_pipeline(
        filter_type=args.filter,
        model_name=args.model,
        feature_model=args.feature_model,
        cluster_algo=args.cluster_algo,
        num_clusters=args.num_clusters,
        skip_download=args.skip_download,
    )


if __name__ == "__main__":
    main()
