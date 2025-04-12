#!/usr/bin/env python3

import os
import argparse
from PIL import Image
import sys
from tqdm import tqdm

# Class ID mapping: Old ID string -> New ID string
label_mapping_dict = {
	"0": "0", "1": "0", "2": "0", "3": "0",  # Animal
	"4": "2",  # Human
	"5": "1", "6": "1", "8": "1",  # Hand
	"7": "3", "18": "3",  # Animal_print
	"9": "4", "10": "4", "11": "4", "12": "4", "15": "4", "16": "4", "17": "4",  # Geometric
	"13": "5", "14": "5"  # Other
}

# Output class IDs to keep after mapping
labels_of_interest = ["0", "1"]  # e.g., Animal, Hand


def parse_yolo_label(txt_file):
	# Reads YOLO label file: returns list of (class_id, xc, yc, w, h)
	if not os.path.exists(txt_file): return []
	boxes = []
	try:
		with open(txt_file, "r") as f:
			for line in f:
				parts = line.strip().split()
				if len(parts) == 5:
					boxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
	except Exception as e:
		print(f"Error parsing label file {txt_file}: {e}", file=sys.stderr)
	return boxes


def convert_yolo_to_abs(class_id, x_c, y_c, w, h, img_w, img_h):
	# Converts YOLO normalized coords to absolute pixel coords (xmin, ymin, xmax, ymax)
	xmin = (x_c - w / 2) * img_w
	ymin = (y_c - h / 2) * img_h
	xmax = (x_c + w / 2) * img_w
	ymax = (y_c + h / 2) * img_h
	return (class_id, xmin, ymin, xmax, ymax)


def convert_abs_to_yolo(class_id, xmin, ymin, xmax, ymax, tile_w, tile_h):
	# Converts absolute coords relative to tile origin to YOLO normalized coords
	if tile_w <= 0 or tile_h <= 0: return None
	box_w = xmax - xmin
	box_h = ymax - ymin
	x_center = xmin + box_w / 2
	y_center = ymin + box_h / 2

	rel_xc = max(0.0, min(1.0, x_center / tile_w))
	rel_yc = max(0.0, min(1.0, y_center / tile_h))
	rel_w = max(0.0, min(1.0, box_w / tile_w))
	rel_h = max(0.0, min(1.0, box_h / tile_h))

	if rel_w <= 1e-6 or rel_h <= 1e-6: return None  # Avoid zero-dimension boxes
	return (class_id, rel_xc, rel_yc, rel_w, rel_h)


def tile_image_and_labels_with_overlap(
		img_path, lbl_path, out_img_dir, out_lbl_dir,
		tile_size, overlap, allow_partial_tiles, skip_empty_tiles
):
	# Tiles a single image and its labels with overlap
	try:
		img_name = os.path.splitext(os.path.basename(img_path))[0]
		image = Image.open(img_path).convert("RGB")
		W, H = image.size
	except Exception as e:
		print(f"Warning: Skipping image {img_path} due to error: {e}", file=sys.stderr)
		return

	orig_boxes = parse_yolo_label(lbl_path)
	if not os.path.exists(lbl_path):
		print(f"Warning: Label file not found {lbl_path}, tiling image without labels.", file=sys.stderr)

	# Map labels and convert to absolute coordinates
	abs_boxes = []
	for (old_cid, x_c, y_c, w_n, h_n) in orig_boxes:
		new_label_str = label_mapping_dict.get(str(old_cid), None)
		if new_label_str in labels_of_interest:
			final_cid = int(new_label_str)
			_, x_min, y_min, x_max, y_max = convert_yolo_to_abs(final_cid, x_c, y_c, w_n, h_n, W, H)
			if x_max > x_min and y_max > y_min:  # Check validity
				abs_boxes.append((final_cid, x_min, y_min, x_max, y_max))

	# Calculate step size for tiling
	step = tile_size - overlap

	# Iterate through tile start coordinates
	for y0 in range(0, H, step):
		for x0 in range(0, W, step):
			x1 = min(W, x0 + tile_size)
			y1 = min(H, y0 + tile_size)
			tw = x1 - x0
			th = y1 - y0

			if tw <= 0 or th <= 0: continue  # Skip invalid dimension tiles

			# Handle partial tiles at edges
			is_partial = (tw < tile_size or th < tile_size)
			if is_partial and not allow_partial_tiles: continue

			# Find labels intersecting with this tile
			tile_boxes = []
			for (cid, axmin, aymin, axmax, aymax) in abs_boxes:
				inter_xmin = max(x0, axmin)
				inter_ymin = max(y0, aymin)
				inter_xmax = min(x1, axmax)
				inter_ymax = min(y1, aymax)

				if inter_xmax > inter_xmin and inter_ymax > inter_ymin:  # Check intersection
					local_xmin, local_ymin = inter_xmin - x0, inter_ymin - y0
					local_xmax, local_ymax = inter_xmax - x0, inter_ymax - y0
					yolo_box = convert_abs_to_yolo(cid, local_xmin, local_ymin, local_xmax, local_ymax, tw, th)
					if yolo_box:
						tile_boxes.append(yolo_box)

			if skip_empty_tiles and not tile_boxes: continue  # Skip empty tiles if requested

			# Save tile image and corresponding label file (if boxes exist)
			try:
				tile_img = image.crop((x0, y0, x1, y1))
				tile_fname = f"{img_name}_x{x0}_y{y0}_s{tile_size}.jpg"
				tile_img_path = os.path.join(out_img_dir, tile_fname)
				tile_img.save(tile_img_path, quality=95)

				if tile_boxes:
					lbl_fname = f"{img_name}_x{x0}_y{y0}_s{tile_size}.txt"
					tile_lbl_path = os.path.join(out_lbl_dir, lbl_fname)
					with open(tile_lbl_path, "w") as lf:
						for (cid2, xcn, ycn, wn, hn) in tile_boxes:
							lf.write(f"{cid2} {xcn:.6f} {ycn:.6f} {wn:.6f} {hn:.6f}\n")
			except Exception as e:
				print(f"Error saving tile x{x0}_y{y0} for {img_name}: {e}", file=sys.stderr)


def write_grouped_classes(output_base_dir):
	# Writes the grouped_classes.txt file based on labels_of_interest
	# Determines path relative to output_base_dir
	parent_dir = os.path.dirname(output_base_dir)
	target_path = os.path.join(parent_dir, "grouped_classes.txt")

	# Fallback if structure isn't as expected (e.g., output_base is root)
	if not os.path.basename(parent_dir).startswith("tiles"):
		target_path = os.path.join(output_base_dir, "grouped_classes.txt")

	print(f"Attempting to write grouped_classes.txt to: {target_path}")
	os.makedirs(os.path.dirname(target_path), exist_ok=True)

	try:
		# Simple hardcoded version based on common use case
		class_content = "0 Animal\n1 Hand\n"
		with open(target_path, "w") as gf:
			gf.write(class_content)
		print(f"Created/Updated grouped_classes.txt at: {target_path}")
	except Exception as e:
		print(f"Error writing grouped_classes.txt to {target_path}: {e}", file=sys.stderr)


def main():
	parser = argparse.ArgumentParser(description="Tile images with overlap into a YOLO dataset.")
	parser.add_argument("--input_images", required=True, help="Directory of large original images.")
	parser.add_argument("--input_labels", required=True, help="Directory of YOLO labels for original images.")
	parser.add_argument("--output_base", required=True, help="Output directory for the tiled dataset.")
	parser.add_argument("--tile_size", type=int, default=512, help="Tile dimension (pixels).")
	parser.add_argument("--overlap", type=int, default=100, help="Overlap between tiles in pixels (default: 100).")
	parser.add_argument("--allow_partial_tiles", action="store_true", help="Produce partial tiles at edges.")
	parser.add_argument("--skip_empty_tiles", action="store_true", help="Skip saving tiles with no target labels.")
	parser.add_argument("--image_ext", default=".jpg", help="Input image file extension.")
	args = parser.parse_args()

	if args.overlap < 0 or args.overlap >= args.tile_size:
		print(f"Error: Overlap ({args.overlap}) invalid for tile size ({args.tile_size}).", file=sys.stderr)
		sys.exit(1)

	out_img_dir = os.path.join(args.output_base, "images")
	out_lbl_dir = os.path.join(args.output_base, "labels")
	os.makedirs(out_img_dir, exist_ok=True)
	os.makedirs(out_lbl_dir, exist_ok=True)

	valid_exts = tuple(ext.strip().lower() for ext in args.image_ext.split(','))
	if ".jpg" in valid_exts and ".jpeg" not in valid_exts: valid_exts += (".jpeg",)

	try:
		img_files = sorted([f for f in os.listdir(args.input_images) if f.lower().endswith(valid_exts)])
	except FileNotFoundError:
		print(f"Error: Input image directory not found: {args.input_images}", file=sys.stderr)
		sys.exit(1)

	if not img_files:
		print(f"Error: No images found with extension(s) '{','.join(valid_exts)}' in '{args.input_images}'.",
			  file=sys.stderr)
		sys.exit(1)

	print(f"Found {len(img_files)} images. Starting tiling (size={args.tile_size}, overlap={args.overlap})...")

	# Process images
	for img_file in tqdm(img_files, desc="Tiling images"):
		base_name = os.path.splitext(img_file)[0]
		img_path = os.path.join(args.input_images, img_file)
		lbl_path = os.path.join(args.input_labels, base_name + ".txt")

		tile_image_and_labels_with_overlap(
			img_path, lbl_path, out_img_dir, out_lbl_dir,
			args.tile_size, args.overlap,
			args.allow_partial_tiles, args.skip_empty_tiles
		)

	# Write grouped classes file
	write_grouped_classes(args.output_base)

	print(f"\nTiling completed. Tiled dataset is in: {args.output_base}")


if __name__ == "__main__":
	main()
