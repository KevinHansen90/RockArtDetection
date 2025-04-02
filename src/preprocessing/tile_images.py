#!/usr/bin/env python3
"""
tile_images.py

1) Reads large images + YOLO labels from a specified directory.
2) Tiles them into 512x512 patches.
3) Maps your old class IDs onto new YOLO classes (e.g., 0=Animal, 1=Hand).
4) Saves tiles + .txt labels, optionally skipping empty tiles if --skip_empty_tiles is used.
5) Writes data/grouped_classes.txt (NOT in data/tiles/base, but in data/).

IMPORTANT:
- The output label files will contain 0-indexed class IDs (e.g., "0" for Animal, "1" for Hand).
- Our CustomYOLODataset adds +1 to each label (so that background=0).
- Therefore, if grouped_classes.txt contains:
      0 Animal
      1 Hand
  then the model will see labels as 1 (Animal) and 2 (Hand), with 0 reserved for background.
"""

import os
import math
import argparse
from PIL import Image

# 1) Mapping from old IDs => new YOLO IDs.
label_mapping_dict = {
	"0": "0",  # Animal
	"1": "0",
	"2": "0",
	"3": "0",
	"4": "2",  # e.g. Human
	"5": "1",  # Hand
	"6": "1",
	"7": "3",  # Animal_print
	"8": "1",  # Hand
	"9": "4",  # Geometric
	"10": "4",
	"11": "4",
	"12": "4",
	"13": "5",  # Other
	"14": "5",
	"15": "4",
	"16": "4",
	"17": "4",
	"18": "3"  # Animal_print
}

# We only keep YOLO classes "0" (Animal) and "1" (Hand).
labels_of_interest = ["0", "1"]


def parse_yolo_label(txt_file):
	"""
    Reads YOLO label => list of (class_id, x_center, y_center, w, h).
    """
	if not os.path.exists(txt_file):
		return []
	boxes = []
	with open(txt_file, "r") as f:
		for line in f:
			parts = line.strip().split()
			if len(parts) == 5:
				c_id, x_c, y_c, w, h = parts
				boxes.append((int(c_id), float(x_c), float(y_c), float(w), float(h)))
	return boxes


def convert_yolo_to_abs(class_id, x_c, y_c, w, h, img_w, img_h):
	"""YOLO [0..1] => absolute coords (xmin, ymin, xmax, ymax)."""
	xmin = (x_c - w / 2) * img_w
	ymin = (y_c - h / 2) * img_h
	xmax = (x_c + w / 2) * img_w
	ymax = (y_c + h / 2) * img_h
	return (class_id, xmin, ymin, xmax, ymax)


def convert_abs_to_yolo(class_id, xmin, ymin, xmax, ymax, tile_w, tile_h):
	"""Absolute coords => YOLO [0..1] relative to tile."""
	box_w = xmax - xmin
	box_h = ymax - ymin
	x_center = xmin + box_w / 2
	y_center = ymin + box_h / 2
	return (
		class_id,
		x_center / tile_w,
		y_center / tile_h,
		box_w / tile_w,
		box_h / tile_h
	)


def get_valid_tiles(W, H, tile_size, allow_partial_tiles):
	"""
    Generator that yields (x0, y0, x1, y1) for valid tiles.
    Handles partial-tile logic and ensures tile dimensions > 0.
    """
	if allow_partial_tiles:
		nx = math.ceil(W / tile_size)
		ny = math.ceil(H / tile_size)
	else:
		nx = W // tile_size
		ny = H // tile_size

	for row in range(ny):
		y0 = row * tile_size
		y1 = min(H, y0 + tile_size)
		if y0 >= H:
			break

		for col in range(nx):
			x0 = col * tile_size
			x1 = min(W, x0 + tile_size)
			if x0 >= W:
				break

			tw = x1 - x0
			th = y1 - y0

			# skip if tile is 0-dim
			if tw <= 0 or th <= 0:
				continue

			# skip partial if not allowed
			if (not allow_partial_tiles) and (tw < tile_size or th < tile_size):
				continue

			yield (x0, y0, x1, y1)


def tile_image_and_labels(img_path, lbl_path, out_img_dir, out_lbl_dir,
						  tile_size=512,
						  allow_partial_tiles=False,
						  skip_empty_tiles=False):
	"""
    Tiles the image into tile_size x tile_size patches.
    Maps old class IDs => new YOLO IDs => absolute coords => tile coords.
    Optionally skip tiles with zero bounding boxes if skip_empty_tiles=True.
    """
	img_name = os.path.splitext(os.path.basename(img_path))[0]
	image = Image.open(img_path).convert("RGB")
	W, H = image.size

	orig_boxes = parse_yolo_label(lbl_path)

	# Convert old IDs => new YOLO IDs => absolute coords
	abs_boxes = []
	for (old_cid, x_c, y_c, w_n, h_n) in orig_boxes:
		# Only proceed if old_cid is mapped and resulting class is of interest
		new_label_str = label_mapping_dict.get(str(old_cid), None)
		if new_label_str in labels_of_interest:
			final_cid = int(new_label_str)  # 0 => Animal, 1 => Hand
			c_id, x_min, y_min, x_max, y_max = convert_yolo_to_abs(
				final_cid, x_c, y_c, w_n, h_n, W, H
			)
			abs_boxes.append((c_id, x_min, y_min, x_max, y_max))

	# Skip this image entirely if no boxes remain
	if len(abs_boxes) == 0:
		return

	# Generate all valid tiles
	for (x0, y0, x1, y1) in get_valid_tiles(W, H, tile_size, allow_partial_tiles):
		tile_img = image.crop((x0, y0, x1, y1))
		tw, th = tile_img.size

		tile_boxes = []
		for (cid, axmin, aymin, axmax, aymax) in abs_boxes:
			cxmin = max(x0, axmin)
			cymin = max(y0, aymin)
			cxmax = min(x1, axmax)
			cymax = min(y1, aymax)

			if cxmax <= cxmin or cymax <= cymin:
				continue

			lxmin = cxmin - x0
			lymin = cymin - y0
			lxmax = cxmax - x0
			lymax = cymax - y0

			# Convert absolute => YOLO relative (boxes remain 0-indexed here)
			new_box = convert_abs_to_yolo(cid, lxmin, lymin, lxmax, lymax, tw, th)
			_, xcn, ycn, wn, hn = new_box

			# Filter boxes that fall completely outside
			if wn <= 0 or hn <= 0:
				continue
			if not (0 <= xcn <= 1 and 0 <= ycn <= 1 and 0 <= wn <= 1 and 0 <= hn <= 1):
				continue

			tile_boxes.append(new_box)

		# skip empty if requested
		if skip_empty_tiles and len(tile_boxes) == 0:
			continue

		# Write tile image & label
		tile_fname = f"{img_name}_r{y0 // tile_size}_c{x0 // tile_size}.jpg"
		tile_img_path = os.path.join(out_img_dir, tile_fname)
		tile_img.save(tile_img_path, quality=95)

		lbl_fname = f"{img_name}_r{y0 // tile_size}_c{x0 // tile_size}.txt"
		tile_lbl_path = os.path.join(out_lbl_dir, lbl_fname)
		with open(tile_lbl_path, "w") as lf:
			for (cid2, xcn, ycn, wn, hn) in tile_boxes:
				lf.write(f"{cid2} {xcn:.6f} {ycn:.6f} {wn:.6f} {hn:.6f}\n")


def main():
	parser = argparse.ArgumentParser(
		description="Tile images => YOLO dataset (Animal=0, Hand=1). Writes grouped_classes.txt to data/."
	)
	parser.add_argument("--input_images", required=True, help="Dir of large original images")
	parser.add_argument("--input_labels", required=True, help="Dir of YOLO label .txt")
	parser.add_argument("--output_base", required=True,
						help="Output directory for the base tiled dataset, e.g. data/tiles/base")
	parser.add_argument("--tile_size", type=int, default=512, help="Tile dimension")
	parser.add_argument("--allow_partial_tiles", action="store_true", default=False,
						help="Produce partial tiles at right/bottom edges.")
	parser.add_argument("--skip_empty_tiles", action="store_true", default=False,
						help="If true, skip saving tiles that have zero bounding boxes.")
	args = parser.parse_args()

	out_img_dir = os.path.join(args.output_base, "images")
	out_lbl_dir = os.path.join(args.output_base, "labels")
	os.makedirs(out_img_dir, exist_ok=True)
	os.makedirs(out_lbl_dir, exist_ok=True)

	# Collect valid image files
	valid_exts = (".jpg", ".jpeg", ".png")
	img_files = [f for f in os.listdir(args.input_images) if f.lower().endswith(valid_exts)]
	img_files.sort()

	# Perform tiling for each image
	for img_file in img_files:
		base_name = os.path.splitext(img_file)[0]
		img_path = os.path.join(args.input_images, img_file)
		lbl_path = os.path.join(args.input_labels, base_name + ".txt")

		tile_image_and_labels(
			img_path=img_path,
			lbl_path=lbl_path,
			out_img_dir=out_img_dir,
			out_lbl_dir=out_lbl_dir,
			tile_size=args.tile_size,
			allow_partial_tiles=args.allow_partial_tiles,
			skip_empty_tiles=args.skip_empty_tiles
		)

	# Write grouped_classes.txt in data/, not data/tiles/base
	data_dir = os.path.dirname(os.path.dirname(args.output_base))
	final_grouped_path = os.path.join(data_dir, "grouped_classes.txt")

	with open(final_grouped_path, "w") as gf:
		gf.write("0 Animal\n")
		gf.write("1 Hand\n")

	print("Tiling completed. Base dataset is in:", args.output_base)
	print("Created grouped_classes.txt (YOLO class IDs) at:", final_grouped_path)


if __name__ == "__main__":
	main()
