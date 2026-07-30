#!/usr/bin/env python3
"""
Fast Single-Upload GCS Data Seeder for RockArtDetection Cloud Jobs.
Creates local dataset structure and performs a single recursive upload to GCS.
"""

import os
import random
import shutil
import tempfile
import subprocess

BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "rockart-data")
CLASSES = ["Animal", "Hand"]
SUBSETS = ["base", "bilateral", "unsharp", "laplacian", "clahe"]
SPLITS = ["train", "val", "test"]


def seed_gcs():
    root = tempfile.mkdtemp()
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)

    # 1. Classes file
    classes_path = os.path.join(data_dir, "grouped_classes.txt")
    with open(classes_path, "w") as f:
        f.write("\n".join(CLASSES) + "\n")

    # 2. Minimal JPEG header + raw data (or PPM)
    ppm_header = b"P6\n256 256\n255\n"
    dummy_pixel_data = bytes([random.randint(50, 200) for _ in range(256 * 256 * 3)])
    image_bytes = ppm_header + dummy_pixel_data

    # Generate full directory tree locally first
    for subset in SUBSETS:
        for split in SPLITS:
            img_dir = os.path.join(data_dir, "tiles", subset, split, "images")
            lbl_dir = os.path.join(data_dir, "tiles", subset, split, "labels")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)

            num_samples = 6 if split == "train" else 2
            for i in range(num_samples):
                img_name = f"tile_{subset}_{split}_{i:03d}.jpg"
                lbl_name = f"tile_{subset}_{split}_{i:03d}.txt"

                with open(os.path.join(img_dir, img_name), "wb") as f:
                    f.write(image_bytes)

                with open(os.path.join(lbl_dir, lbl_name), "w") as f:
                    f.write("0 0.5 0.5 0.3 0.3\n1 0.2 0.3 0.15 0.2\n")

    print(f"[*] Single recursive upload of dataset to gs://{BUCKET_NAME}/data/ ...")
    subprocess.run(f"gcloud storage cp -r {data_dir}/* gs://{BUCKET_NAME}/data/", shell=True, check=True)
    shutil.rmtree(root)
    print(f"[+] Successfully seeded gs://{BUCKET_NAME}/data/")


if __name__ == "__main__":
    seed_gcs()
