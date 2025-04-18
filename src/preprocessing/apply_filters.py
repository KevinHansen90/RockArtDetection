#!/usr/bin/env python3

import os
import sys
import argparse
import shutil
from PIL import Image, ImageFilter
import cv2
import numpy as np
from tqdm import tqdm


def apply_bilateral(pil_img, d=9, sigmaColor=75, sigmaSpace=75):
    """
    Bilateral filter using OpenCV. d: pixel neighborhood diameter.
    """
    np_img = np.array(pil_img)
    filtered = cv2.bilateralFilter(np_img, d, sigmaColor, sigmaSpace)
    return Image.fromarray(filtered)


def apply_unsharp(pil_img, radius=2, percent=150, threshold=3):
    """
    Unsharp mask filter using PIL's ImageFilter.
    """
    return pil_img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


def apply_laplacian(pil_img):
    """
    Laplacian filter: grayscale -> laplacian -> back to RGB.
    """
    np_img = np.array(pil_img)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = cv2.convertScaleAbs(lap)
    lap_3ch = cv2.cvtColor(lap, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(lap_3ch)


def apply_clahe(pil_img):
    """
    CLAHE in LAB color space to enhance contrast.
    """
    np_img = np.array(pil_img)
    lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    clahe_img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(clahe_img)


def copy_labels(src_labels_dir, dst_labels_dir):
    """
    Recursively copy label files (.txt) from src to dst, preserving directory structure.
    """
    if os.path.exists(dst_labels_dir):
        shutil.rmtree(dst_labels_dir)
    shutil.copytree(src_labels_dir, dst_labels_dir)


def process_subset(base_dir, output_dir, subset, filter_fn):
    """
    Apply filter_fn to all images in base_dir/subset/images,
    copy labels from base_dir/subset/labels to output_dir/subset/labels
    """
    src_img_dir  = os.path.join(base_dir, subset, 'images')
    src_lbl_dir  = os.path.join(base_dir, subset, 'labels')
    dst_img_dir  = os.path.join(output_dir, subset, 'images')
    dst_lbl_dir  = os.path.join(output_dir, subset, 'labels')

    # Create output dirs
    os.makedirs(dst_img_dir, exist_ok=True)
    copy_labels(src_lbl_dir, dst_lbl_dir)

    # Process images
    valid_exts = ('.jpg', '.jpeg', '.png')
    img_files = [f for f in os.listdir(src_img_dir) if f.lower().endswith(valid_exts)]
    if not img_files:
        print(f"Warning: No images found in {src_img_dir}", file=sys.stderr)
        return

    print(f"[{subset}] Applying filter to {len(img_files)} images...")
    for img_file in tqdm(img_files, desc=f"Filtering {subset}"):
        src_path = os.path.join(src_img_dir, img_file)
        dst_path = os.path.join(dst_img_dir, img_file)
        try:
            img = Image.open(src_path).convert('RGB')
            filtered = filter_fn(img)
            filtered.save(dst_path, quality=95)
        except Exception as e:
            print(f"Error processing {src_path}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Apply image filters to a train/val/test dataset tree.")
    parser.add_argument('--base_dir',    required=True,
                        help='Root dataset dir containing train/, val/, test/')
    parser.add_argument('--output_dir',  required=True,
                        help='Destination root for filtered dataset')
    parser.add_argument('--filter_type', required=True,
                        choices=['bilateral','unsharp','laplacian','clahe'],
                        help='Filter to apply')
    args = parser.parse_args()

    # Select filter function
    fns = {
        'bilateral': apply_bilateral,
        'unsharp':   apply_unsharp,
        'laplacian': apply_laplacian,
        'clahe':     apply_clahe
    }
    filter_fn = fns[args.filter_type]

    # Copy entire folder structure except images, apply filter to images
    subsets = ['train','val','test']
    for subset in subsets:
        print(f"Processing subset: {subset}")
        process_subset(args.base_dir, args.output_dir, subset, filter_fn)

    print(f"Filter '{args.filter_type}' applied. Output at: {args.output_dir}")


if __name__ == '__main__':
    main()

