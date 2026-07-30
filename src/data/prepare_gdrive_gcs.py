#!/usr/bin/env python3
"""
Real Dataset Ingestion & Preprocessing Script for GCS Sync.
Downloads 683 real images from Google Drive, tiles, applies filters, and syncs to GCS.
"""

import os
import shutil
import subprocess
from src.data.download_drive import download_from_gdrive
from src.data.tile_images import tile_image_and_labels_with_overlap, write_grouped_classes
from src.data.split_dataset import copy_image_and_label
from src.data.apply_filters import process_subset, apply_bilateral, apply_unsharp, apply_laplacian, apply_clahe

GDRIVE_URL = "https://drive.google.com/drive/u/0/folders/1JU5tohaRw7Rm83S9uUK9KazIPLRebl1x"
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "your-gcs-bucket-name")
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
TILES_BASE = os.path.join(DATA_DIR, "tiles", "base")


def prepare_and_sync():
    print("==================================================")
    print("  REAL DATASET INGESTION & GCS PREPROCESSING      ")
    print("==================================================")

    # 1. Download Google Drive dataset if not present
    raw_images_dir = os.path.join(RAW_DIR, "images")
    if not os.path.exists(raw_images_dir) or len(os.listdir(raw_images_dir)) == 0:
        print("[*] Downloading real dataset from Google Drive...")
        download_from_gdrive(GDRIVE_URL, RAW_DIR, is_folder=True)
    else:
        print(f"[*] Found {len(os.listdir(raw_images_dir))} raw images in '{raw_images_dir}'. Skipping download.")

    # 2. Tile raw images with overlap
    out_img_dir = os.path.join(TILES_BASE, "images")
    out_lbl_dir = os.path.join(TILES_BASE, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    raw_lbls_dir = os.path.join(RAW_DIR, "labels")
    if os.path.exists(raw_images_dir):
        print("[*] Tiling images into 512x512 tiles with 100px overlap...")
        for img_file in os.listdir(raw_images_dir):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                base_name = os.path.splitext(img_file)[0]
                img_path = os.path.join(raw_images_dir, img_file)
                lbl_path = os.path.join(raw_lbls_dir, base_name + ".txt")
                tile_image_and_labels_with_overlap(
                    img_path, lbl_path, out_img_dir, out_lbl_dir,
                    tile_size=512, overlap=100, allow_partial_tiles=True, skip_empty_tiles=True
                )
        write_grouped_classes(TILES_BASE)

    # 3. Create Train / Val / Test Splits (80/5/15)
    print("[*] Splitting dataset into train/val/test...")
    all_imgs = [f for f in os.listdir(out_img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    import random
    random.seed(42)
    random.shuffle(all_imgs)

    n_total = len(all_imgs)
    n_train = int(n_total * 0.80)
    n_val = int(n_total * 0.05)

    splits = {
        "train": all_imgs[:n_train],
        "val": all_imgs[n_train:n_train + n_val],
        "test": all_imgs[n_train + n_val:]
    }

    for split_name, img_list in splits.items():
        s_img_dir = os.path.join(TILES_BASE, split_name, "images")
        s_lbl_dir = os.path.join(TILES_BASE, split_name, "labels")
        os.makedirs(s_img_dir, exist_ok=True)
        os.makedirs(s_lbl_dir, exist_ok=True)

        for img_f in img_list:
            copy_image_and_label(
                img_f,
                out_img_dir,
                out_lbl_dir,
                s_img_dir,
                s_lbl_dir
            )

    # 4. Apply 4 Filter Preprocessing Variants
    filter_fns = {
        "bilateral": apply_bilateral,
        "unsharp": apply_unsharp,
        "laplacian": apply_laplacian,
        "clahe": apply_clahe,
    }

    for f_name, f_func in filter_fns.items():
        print(f"[*] Preprocessing dataset filter subset '{f_name}'...")
        f_dir = os.path.join(DATA_DIR, "tiles", f_name)
        for subset in ["train", "val", "test"]:
            if os.path.exists(os.path.join(TILES_BASE, subset)):
                process_subset(TILES_BASE, f_dir, subset, f_func)

    # 5. Sync to GCS
    print(f"[*] Uploading preprocessed real dataset tree to gs://{BUCKET_NAME}/data/...")
    grouped_classes_file = os.path.join(DATA_DIR, "grouped_classes.txt")
    if not os.path.exists(grouped_classes_file):
        grouped_classes_file = os.path.join(TILES_BASE, "grouped_classes.txt")

    if os.path.exists(grouped_classes_file):
        subprocess.run(f"gcloud storage cp {grouped_classes_file} gs://{BUCKET_NAME}/data/grouped_classes.txt", shell=True, check=True)

    subprocess.run(f"gcloud storage cp -r {DATA_DIR}/tiles gs://{BUCKET_NAME}/data/", shell=True, check=True)
    print(f"[+] Real dataset successfully uploaded to gs://{BUCKET_NAME}/data/")


if __name__ == "__main__":
    prepare_and_sync()
