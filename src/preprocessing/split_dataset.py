#!/usr/bin/env python3
"""
split_dataset.py

Splits a dataset of images/labels (in YOLO format) into train, val, and test subsets.

Example usage:
  python split_dataset.py \
    --input_dir data/tiles/base \
    --output_dir data/tiles/base \
    --train_ratio 0.75 \
    --val_ratio 0.05 \
    --test_ratio 0.20 \
    --seed 42
"""

import os
import random
import argparse
import shutil


def main():
	parser = argparse.ArgumentParser(description="Split YOLO dataset into train/val/test.")
	parser.add_argument("--input_dir", required=True,
						help="Directory containing 'images' and 'labels' subfolders.")
	parser.add_argument("--output_dir", required=True,
						help="Directory where the split subfolders (train/val/test) will be created.")
	parser.add_argument("--train_ratio", type=float, default=0.75,
						help="Fraction of dataset to use for training.")
	parser.add_argument("--val_ratio", type=float, default=0.05,
						help="Fraction of dataset to use for validation.")
	parser.add_argument("--test_ratio", type=float, default=0.20,
						help="Fraction of dataset to use for testing.")
	parser.add_argument("--seed", type=int, default=42,
						help="Random seed for reproducibility.")
	args = parser.parse_args()

	input_images_dir = os.path.join(args.input_dir, "images")
	input_labels_dir = os.path.join(args.input_dir, "labels")

	# Output subdirs: train, val, test (each with images + labels)
	train_img_dir = os.path.join(args.output_dir, "train", "images")
	train_lbl_dir = os.path.join(args.output_dir, "train", "labels")
	val_img_dir = os.path.join(args.output_dir, "val", "images")
	val_lbl_dir = os.path.join(args.output_dir, "val", "labels")
	test_img_dir = os.path.join(args.output_dir, "test", "images")
	test_lbl_dir = os.path.join(args.output_dir, "test", "labels")

	# Create them
	for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir, test_img_dir, test_lbl_dir]:
		os.makedirs(d, exist_ok=True)

	# Basic ratio check
	total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
	if not (0.99 <= total_ratio <= 1.01):
		raise ValueError(f"Train/val/test ratios must sum to ~1.0. Got {total_ratio}")

	# Gather all image filenames
	valid_exts = (".jpg", ".jpeg", ".png")
	all_images = [f for f in os.listdir(input_images_dir) if f.lower().endswith(valid_exts)]
	all_images.sort()

	# Shuffle with seed
	random.seed(args.seed)
	random.shuffle(all_images)

	total_count = len(all_images)
	train_count = int(total_count * args.train_ratio)
	val_count = int(total_count * args.val_ratio)
	test_count = total_count - train_count - val_count  # remainder

	train_images = all_images[:train_count]
	val_images = all_images[train_count: train_count + val_count]
	test_images = all_images[train_count + val_count:]

	print(f"Total images: {total_count}")
	print(f" -> Train: {len(train_images)}")
	print(f" -> Val:   {len(val_images)}")
	print(f" -> Test:  {len(test_images)}")

	# Utility to copy image + label
	def copy_image_and_label(img_file, source_img_dir, source_lbl_dir, dest_img_dir, dest_lbl_dir):
		# Copy the image
		src_img_path = os.path.join(source_img_dir, img_file)
		dst_img_path = os.path.join(dest_img_dir, img_file)
		shutil.copy2(src_img_path, dst_img_path)

		# Copy the corresponding label file if exists
		base_name = os.path.splitext(img_file)[0]
		lbl_file = base_name + ".txt"
		src_lbl_path = os.path.join(source_lbl_dir, lbl_file)
		if os.path.exists(src_lbl_path):
			dst_lbl_path = os.path.join(dest_lbl_dir, lbl_file)
			shutil.copy2(src_lbl_path, dst_lbl_path)

	# Copy train images/labels
	for img_file in train_images:
		copy_image_and_label(img_file, input_images_dir, input_labels_dir, train_img_dir, train_lbl_dir)
	# Copy val
	for img_file in val_images:
		copy_image_and_label(img_file, input_images_dir, input_labels_dir, val_img_dir, val_lbl_dir)
	# Copy test
	for img_file in test_images:
		copy_image_and_label(img_file, input_images_dir, input_labels_dir, test_img_dir, test_lbl_dir)

	print("Dataset split completed!")
	print(f"Train: {train_img_dir}, {train_lbl_dir}")
	print(f"Val:   {val_img_dir}, {val_lbl_dir}")
	print(f"Test:  {test_img_dir}, {test_lbl_dir}")


if __name__ == "__main__":
	main()
