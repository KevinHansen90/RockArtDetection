#!/usr/bin/env python3
"""
apply_filters.py

Apply one of several filters (Bilateral, Unsharp, Laplacian, CLAHE) to an existing
train/val/test dataset (images), copying the labels as-is (bounding boxes don't change).

Usage:
  python apply_filters.py \
    --base_dir data/tiles/base \
    --output_dir data/tiles/bilateral \
    --filter_type bilateral

Possible --filter_type values:
  - bilateral
  - unsharp
  - laplacian
  - clahe

This will produce subfolders in the output_dir:
  train/images, train/labels
  val/images, val/labels
  test/images, test/labels
The bounding box .txt files are copied from the base_dir.

Run this script multiple times (once for each filter_type), specifying different output_dir.
"""

import os
import argparse
from PIL import Image, ImageFilter
import cv2
import numpy as np
import shutil


def apply_bilateral(pil_img, d=9, sigmaColor=75, sigmaSpace=75):
	"""
    Bilateral filter using OpenCV. d is the diameter of each pixel neighborhood.
    """
	np_img = np.array(pil_img)
	filtered = cv2.bilateralFilter(np_img, d, sigmaColor, sigmaSpace)
	return Image.fromarray(filtered)


def apply_unsharp(pil_img, radius=2, percent=150, threshold=3):
	"""
    Unsharp mask using PIL's ImageFilter.UnsharpMask.
    """
	return pil_img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


def apply_laplacian(pil_img):
	"""
    Basic Laplacian filter example using OpenCV.
    Convert to grayscale, apply Laplacian, then convert back to 3-channel for consistency.
    """
	np_img = np.array(pil_img)
	gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
	lap = cv2.Laplacian(gray, cv2.CV_64F)
	lap = cv2.convertScaleAbs(lap)
	lap_3ch = cv2.cvtColor(lap, cv2.COLOR_GRAY2RGB)
	return Image.fromarray(lap_3ch)


def apply_clahe(pil_img):
	"""
    Apply CLAHE using OpenCV in LAB color space.
    """
	np_img = np.array(pil_img)
	lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
	l, a, b = cv2.split(lab)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	cl = clahe.apply(l)
	merged = cv2.merge((cl, a, b))
	clahe_img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
	return Image.fromarray(clahe_img)


def main():
	parser = argparse.ArgumentParser(
		description="Apply Bilateral/Unsharp/Laplacian/CLAHE filters to train/val/test data.")
	parser.add_argument("--base_dir", required=True,
						help="Base dataset dir with train/val/test subfolders (each has images & labels).")
	parser.add_argument("--output_dir", required=True,
						help="Output dir for the new filtered dataset, also with train/val/test subfolders.")
	parser.add_argument("--filter_type", required=True,
						choices=["bilateral", "unsharp", "laplacian", "clahe"],
						help="Which filter to apply.")
	args = parser.parse_args()

	filter_type = args.filter_type

	# Decide which function to call
	if filter_type == "bilateral":
		filter_fn = apply_bilateral
	elif filter_type == "unsharp":
		filter_fn = apply_unsharp
	elif filter_type == "laplacian":
		filter_fn = apply_laplacian
	elif filter_type == "clahe":
		filter_fn = apply_clahe
	else:
		raise ValueError(f"Unsupported filter type: {filter_type}")

	# We'll process each subset: train, val, test
	subsets = ["train", "val", "test"]

	for subset in subsets:
		base_img_dir = os.path.join(args.base_dir, subset, "images")
		base_lbl_dir = os.path.join(args.base_dir, subset, "labels")

		out_img_dir = os.path.join(args.output_dir, subset, "images")
		out_lbl_dir = os.path.join(args.output_dir, subset, "labels")
		os.makedirs(out_img_dir, exist_ok=True)
		os.makedirs(out_lbl_dir, exist_ok=True)

		# Get list of images
		valid_exts = (".jpg", ".jpeg", ".png")
		image_files = [f for f in os.listdir(base_img_dir) if f.lower().endswith(valid_exts)]
		image_files.sort()

		# For each image => apply filter
		for img_file in image_files:
			src_img_path = os.path.join(base_img_dir, img_file)
			pil_img = Image.open(src_img_path).convert("RGB")
			filtered_img = filter_fn(pil_img)

			# Save
			out_img_path = os.path.join(out_img_dir, img_file)
			filtered_img.save(out_img_path, quality=95)

			# Copy label
			base_name = os.path.splitext(img_file)[0]
			label_src_path = os.path.join(base_lbl_dir, base_name + ".txt")
			if os.path.exists(label_src_path):
				label_dst_path = os.path.join(out_lbl_dir, base_name + ".txt")
				shutil.copy2(label_src_path, label_dst_path)

		print(f"[{subset}] filter '{filter_type}' done => {args.output_dir}/{subset}")

	print(f"Filter '{filter_type}' applied to train/val/test at: {args.output_dir}")


if __name__ == "__main__":
	main()
