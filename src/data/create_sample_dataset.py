#!/usr/bin/env python3
"""
Create a tiny local 5-sample dataset tree for fast local CLI verification.
"""

import os
import numpy as np
from PIL import Image

SAMPLE_DIR = "data/sample"
SPLITS = ["train", "val", "test"]

for split in SPLITS:
    img_dir = os.path.join(SAMPLE_DIR, split, "images")
    lbl_dir = os.path.join(SAMPLE_DIR, split, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    num = 5 if split == "train" else 2
    for i in range(num):
        img_name = f"sample_{split}_{i:02d}.jpg"
        lbl_name = f"sample_{split}_{i:02d}.txt"

        arr = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(os.path.join(img_dir, img_name), "JPEG")

        with open(os.path.join(lbl_dir, lbl_name), "w") as f:
            f.write("0 0.5 0.5 0.3 0.3\n1 0.2 0.3 0.15 0.2\n")

print(f"[+] Local 5-image sample dataset tree created at '{SAMPLE_DIR}'")
