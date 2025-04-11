# src/preprocessing/crop_motifs.py

import os
import argparse
from PIL import Image, UnidentifiedImageError
import sys
from tqdm import tqdm

# Add project root to sys.path to allow 'from src...' imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # src/preprocessing
_SRC_DIR = os.path.dirname(_SCRIPT_DIR)  # src
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)  # RockArtDetection
if _PROJECT_ROOT not in sys.path:
	sys.path.insert(0, _PROJECT_ROOT)


def parse_yolo_label(txt_file):
	if not os.path.exists(txt_file): return []
	boxes = []
	try:
		with open(txt_file, "r") as f:
			for line in f:
				parts = line.strip().split()
				if len(parts) == 5:
					c_id, x_c, y_c, w, h = parts
					boxes.append((int(c_id), float(x_c), float(y_c), float(w), float(h)))
	except Exception as e:
		print(f"Error parsing label file {txt_file}: {e}", file=sys.stderr)
	return boxes


def yolo_to_absolute_bbox(xc, yc, w, h, img_width, img_height):
	"""Converts YOLO coords (normalized cx, cy, w, h) to absolute pixel bbox [xmin, ymin, xmax, ymax]."""
	box_w = w * img_width
	box_h = h * img_height
	xmin = (xc * img_width) - (box_w / 2)
	ymin = (yc * img_height) - (box_h / 2)
	xmax = xmin + box_w
	ymax = ymin + box_h
	xmin = max(0, int(round(xmin)))
	ymin = max(0, int(round(ymin)))
	xmax = min(img_width, int(round(xmax)))
	ymax = min(img_height, int(round(ymax)))
	return xmin, ymin, xmax, ymax


def crop_resize_and_save_motifs(image_path, label_path, output_dir, target_class_id, resize_dim=None):
	"""Finds motifs, crops them, optionally resizes, and saves them."""
	try:
		img = Image.open(image_path).convert("RGB")
		img_width, img_height = img.size
	except FileNotFoundError:
		# print(f"Warning: Image file not found: {image_path}", file=sys.stderr) # Less verbose
		return 0
	except UnidentifiedImageError:
		print(f"Warning: Cannot read image file (corrupted?): {image_path}", file=sys.stderr)
		return 0

	labels = parse_yolo_label(label_path)
	if not labels: return 0

	count = 0
	base_img_name = os.path.splitext(os.path.basename(image_path))[0]

	for idx, (class_id, xc, yc, w, h) in enumerate(labels):
		if class_id == target_class_id:
			xmin, ymin, xmax, ymax = yolo_to_absolute_bbox(xc, yc, w, h, img_width, img_height)

			if xmax <= xmin or ymax <= ymin:
				# print(f"Warning: Skipping invalid bbox dimensions [{xmin},{ymin},{xmax},{ymax}] for class {target_class_id} in {base_img_name}", file=sys.stderr)
				continue

			try:
				cropped_motif = img.crop((xmin, ymin, xmax, ymax))

				# --- Resizing Logic ---
				if resize_dim and resize_dim > 0:
					# Calculate new size maintaining aspect ratio, targeting the longer side
					original_width, original_height = cropped_motif.size
					aspect_ratio = original_width / original_height

					if original_width >= original_height:  # Width is longer or square
						new_width = resize_dim
						new_height = int(round(new_width / aspect_ratio))
					else:  # Height is longer
						new_height = resize_dim
						new_width = int(round(new_height * aspect_ratio))

					# Ensure dimensions are at least 1 pixel
					new_width = max(1, new_width)
					new_height = max(1, new_height)

					# Use LANCZOS for high-quality downsampling
					resized_motif = cropped_motif.resize((new_width, new_height), Image.Resampling.LANCZOS)
				else:
					resized_motif = cropped_motif  # No resizing needed
				# --- End Resizing Logic ---

				output_filename = f"{base_img_name}_motif_{target_class_id}_{idx}.png"
				output_path = os.path.join(output_dir, output_filename)

				resized_motif.save(output_path, "PNG")
				count += 1
			except Exception as e:
				print(f"Error cropping/resizing/saving motif {idx} from {base_img_name}: {e}", file=sys.stderr)

	return count


def main():
	parser = argparse.ArgumentParser(description="Extract, resize, and save cropped motifs based on YOLO labels.")
	parser.add_argument("--images", required=True, help="Directory containing the original images.")
	parser.add_argument("--labels", required=True,
						help="Directory containing the corresponding YOLO label files (.txt).")
	parser.add_argument("--output", required=True, help="Directory where cropped motif images will be saved.")
	parser.add_argument("--class-id", type=int, required=True,
						help="The integer class ID of the motif to extract (e.g., 0 for 'Animal').")
	parser.add_argument("--image-ext", default=".jpg", help="Extension of the image files (e.g., .jpg, .png).")
	# --- New Argument ---
	parser.add_argument("--resize-dim", type=int, default=None,
						help="Target dimension (pixels) for the *longer* side of the cropped motif. Aspect ratio is maintained. If omitted, no resizing is done.")

	args = parser.parse_args()

	os.makedirs(args.output, exist_ok=True)

	total_cropped_count = 0
	image_files = sorted([f for f in os.listdir(args.images) if f.lower().endswith(args.image_ext)])

	if not image_files:
		print(f"Error: No images found with extension '{args.image_ext}' in directory '{args.images}'", file=sys.stderr)
		sys.exit(1)

	print(f"Found {len(image_files)} images. Starting motif extraction for class ID {args.class_id}...")
	if args.resize_dim:
		print(f"Cropped motifs will be resized so their longer side is {args.resize_dim}px.")

	for img_file in tqdm(image_files, desc="Processing images"):
		img_path = os.path.join(args.images, img_file)
		label_filename = os.path.splitext(img_file)[0] + ".txt"
		label_path = os.path.join(args.labels, label_filename)

		if not os.path.exists(label_path): continue

		cropped_count = crop_resize_and_save_motifs(img_path, label_path, args.output, args.class_id, args.resize_dim)
		total_cropped_count += cropped_count

	print(f"\nExtraction complete. Saved {total_cropped_count} motifs of class ID {args.class_id} to '{args.output}'.")


if __name__ == "__main__":
	main()
