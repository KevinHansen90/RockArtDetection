#!/usr/bin/env python3

import os
import sys
import random
import argparse
import shutil
from tqdm import tqdm


# --- Helper Function ---
def copy_image_and_label(img_file, source_img_dir, source_lbl_dir, dest_img_dir, dest_lbl_dir):
	"""Copies an image and its corresponding label file (if it exists)."""
	try:
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
		return True  # Indicate success
	except Exception as e:
		print(f"\nWarning: Failed to copy {img_file} or its label: {e}", file=sys.stderr)
		return False  # Indicate failure


# --- Main Function ---
def main():
	parser = argparse.ArgumentParser(description="Split YOLO dataset into train/val/test by ratio or number.")
	parser.add_argument("--input_dir", required=True, help="Directory containing 'images' and 'labels' subfolders.")
	parser.add_argument("--output_dir", required=True,
						help="Directory where split subfolders (train/val/test) will be created.")
	parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
	parser.add_argument("--image_ext", default=".jpg", help="Extension of the image files (e.g., .jpg, .png).")

	# Group for splitting method
	split_group = parser.add_mutually_exclusive_group(required=True)
	split_group.add_argument("--use_ratios", action='store_true',
							 help="Split using ratios (requires --train_ratio, etc.).")
	split_group.add_argument("--use_numbers", action='store_true',
							 help="Split using specific numbers (requires --train_num, etc.).")

	# Ratio arguments (used if --use_ratios)
	parser.add_argument("--train_ratio", type=float, default=0.75, help="Fraction for training (if using ratios).")
	parser.add_argument("--val_ratio", type=float, default=0.05, help="Fraction for validation (if using ratios).")
	parser.add_argument("--test_ratio", type=float, default=0.20, help="Fraction for testing (if using ratios).")

	# Number arguments (used if --use_numbers)
	parser.add_argument("--train_num", type=int, help="Number of images for training (if using numbers).")
	parser.add_argument("--val_num", type=int, help="Number of images for validation (if using numbers).")
	parser.add_argument("--test_num", type=int, help="Number of images for testing (if using numbers).")

	args = parser.parse_args()

	input_images_dir = os.path.join(args.input_dir, "images")
	input_labels_dir = os.path.join(args.input_dir, "labels")

	# Define output paths
	paths = {}
	for split in ["train", "val", "test"]:
		paths[f"{split}_img"] = os.path.join(args.output_dir, split, "images")
		paths[f"{split}_lbl"] = os.path.join(args.output_dir, split, "labels")
		os.makedirs(paths[f"{split}_img"], exist_ok=True)
		os.makedirs(paths[f"{split}_lbl"], exist_ok=True)

	# Gather and shuffle image filenames
	valid_exts = tuple(ext.strip().lower() for ext in args.image_ext.split(','))
	if ".jpg" in valid_exts and ".jpeg" not in valid_exts: valid_exts += (".jpeg",)

	try:
		all_images = sorted([f for f in os.listdir(input_images_dir) if f.lower().endswith(valid_exts)])
	except FileNotFoundError:
		print(f"Error: Input image directory not found: {input_images_dir}", file=sys.stderr)
		sys.exit(1)

	if not all_images:
		print(f"Error: No images found with extension(s) '{','.join(valid_exts)}' in '{input_images_dir}'.",
			  file=sys.stderr)
		sys.exit(1)

	random.seed(args.seed)
	random.shuffle(all_images)
	total_count = len(all_images)
	print(f"Found {total_count} total images.")

	# Determine split counts based on chosen method
	train_count, val_count, test_count = 0, 0, 0
	if args.use_ratios:
		total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
		if not (0.99 <= total_ratio <= 1.01):
			raise ValueError(f"Train/val/test ratios must sum to ~1.0. Got {total_ratio}")
		train_count = int(total_count * args.train_ratio)
		val_count = int(total_count * args.val_ratio)
		test_count = total_count - train_count - val_count  # Remainder
	elif args.use_numbers:
		if args.train_num is None or args.val_num is None:
			raise ValueError("--train_num and --val_num must be specified when using --use_numbers.")
		train_count = args.train_num
		val_count = args.val_num
		# Test count can be specified or use the remainder
		if args.test_num is not None:
			test_count = args.test_num
			if train_count + val_count + test_count > total_count:
				raise ValueError(
					f"Sum of train ({train_count}), val ({val_count}), and test ({test_count}) numbers exceeds total images ({total_count}).")
		else:
			test_count = total_count - train_count - val_count  # Remainder
			if test_count < 0:
				raise ValueError(
					f"Sum of train ({train_count}) and val ({val_count}) numbers exceeds total images ({total_count}). No images left for test.")
		print("Note: Using specified numbers for splits.")

	# Slice the shuffled list
	train_images = all_images[:train_count]
	val_images = all_images[train_count: train_count + val_count]
	test_images = all_images[
				  train_count + val_count: train_count + val_count + test_count]  # Use explicit end for number split

	print(f" -> Train: {len(train_images)}")
	print(f" -> Val:   {len(val_images)}")
	print(f" -> Test:  {len(test_images)}")
	if len(train_images) + len(val_images) + len(test_images) < total_count:
		print(
			f" -> Note: {total_count - (len(train_images) + len(val_images) + len(test_images))} images were unused due to split numbers/ratios.")

	# Copy files with progress bars
	print("Copying train files...")
	for img_file in tqdm(train_images, desc="Train"):
		copy_image_and_label(img_file, input_images_dir, input_labels_dir, paths["train_img"], paths["train_lbl"])
	print("Copying val files...")
	for img_file in tqdm(val_images, desc="Val"):
		copy_image_and_label(img_file, input_images_dir, input_labels_dir, paths["val_img"], paths["val_lbl"])
	print("Copying test files...")
	for img_file in tqdm(test_images, desc="Test"):
		copy_image_and_label(img_file, input_images_dir, input_labels_dir, paths["test_img"], paths["test_lbl"])

	print("\nDataset split completed!")
	print(f"Train: {paths['train_img']}, {paths['train_lbl']}")
	print(f"Val:   {paths['val_img']}, {paths['val_lbl']}")
	print(f"Test:  {paths['test_img']}, {paths['test_lbl']}")


if __name__ == "__main__":
	main()
