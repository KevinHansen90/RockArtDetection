# src/training/evaluate.py

import torch
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import sys


def draw_boxes(image, boxes, labels, classes, shift_labels=True, color="red"):
	"""
    Draws bounding boxes and labels on a PIL image.

    Args:
        image (PIL.Image.Image): Image to draw on.
        boxes (list or Tensor): List of boxes, each as [x1, y1, x2, y2].
        labels (list or Tensor): List of labels corresponding to boxes.
        classes (list): List of class names.
        shift_labels (bool): If True, subtract 1 from label index (for F-RCNN/RetinaNet).
        color (str): Color for the boxes and text background.
    """
	draw = ImageDraw.Draw(image)
	margin = 3  # Small margin around text

	try:
		# Try loading a common font, adjust path if needed or handle missing font
		font = ImageFont.truetype("arial.ttf", size=15)
	except IOError:
		font = ImageFont.load_default()

	for box, label in zip(boxes, labels):
		# Convert tensor to list/float if needed.
		if isinstance(box, torch.Tensor):
			box = box.tolist()
		if isinstance(label, torch.Tensor):
			label = label.item()

		# Ensure box coordinates are numeric (float or int) for drawing
		try:
			box_coords = [float(coord) for coord in box]
			box_int = [int(round(coord)) for coord in box_coords]  # Use for drawing rectangle
			left, top, _, _ = box_int  # Get top-left corner for text positioning
		except (ValueError, TypeError) as e:
			print(f"Warning: Skipping box due to invalid coordinates: {box}. Error: {e}", file=sys.stderr)
			continue

		# Draw the bounding box rectangle
		draw.rectangle(box_int, outline=color, width=2)

		# Determine the class index based on the model type
		# Ensure label is treated as an integer index
		try:
			label_int = int(round(label))
			class_index = (label_int - 1) if shift_labels else label_int
		except (ValueError, TypeError):
			print(f"Warning: Skipping label drawing due to invalid label: {label}", file=sys.stderr)
			continue  # Skip text drawing for this box

		# Get the class name text
		if 0 <= class_index < len(classes):
			text = classes[class_index]
		else:
			# Handle cases where label is out of bounds (e.g., background class for F-RCNN)
			text = f"UNK_ID={label_int}"
			if class_index == -1 and shift_labels:  # Likely background for F-RCNN/Retina
				text = "BG"  # Optional: Explicitly label background

		# --- Calculate text size ---
		try:
			# Preferred method: Get bounding box of text
			text_bbox = draw.textbbox((0, 0), text, font=font)
			text_width = text_bbox[2] - text_bbox[0]
			text_height = text_bbox[3] - text_bbox[1]
		except AttributeError:
			# Fallback for older Pillow versions or sometimes with default font
			try:
				text_width, text_height = draw.textlength(text, font=font), 10  # Estimate height for default
				# Note: draw.textsize is also deprecated, textlength is better if bbox fails
			except AttributeError:
				print("Warning: Could not determine text size. Text positioning might be off.", file=sys.stderr)
				text_width, text_height = 30, 10  # Arbitrary fallback size

		# --- Draw text background and text (Moved outside the try/except) ---
		if text:  # Only draw if we have valid text
			# Calculate position for background rectangle (above the box)
			rect_y1 = max(0, top - text_height - margin * 2)  # Ensure y1 is not negative
			rect_y2 = max(text_height + margin, top)  # Anchor bottom of rect near box top
			rect_x1 = left
			rect_x2 = left + text_width + margin * 2

			# Draw background rectangle
			draw.rectangle(
				[(rect_x1, rect_y1), (rect_x2, rect_y2)],
				fill=color
			)
			# Draw text on top of the background rectangle
			# Adjust text position to be within the background rect
			text_x = rect_x1 + margin
			text_y = rect_y1 + margin // 2  # Center text vertically within the rect background
			draw.text((text_x, text_y), text, fill="white", font=font)

	return image


def evaluate_and_visualize(model, test_loader, classes, device, output_path, threshold=0.5, model_type="fasterrcnn"):
	"""
    Evaluates the model on the test set and saves an image visualizing
    ground truth vs predictions for some samples.
    """
	model.eval()
	model.to(device)

	# Determine if labels need shifting based on model type
	# shift_labels is True for models where class 0 is background (FasterRCNN, RetinaNet)
	# shift_labels is False for models where class 0 is the first object (DETR)
	shift_labels = (model_type.lower() != "deformable_detr")

	visualizations = []
	num_samples_to_visualize = min(len(test_loader), 10)  # Visualize up to 10 samples

	# Use enumerate to limit the number of visualized samples easily
	for i, test_data in enumerate(tqdm(test_loader, desc="Evaluating on test set", leave=False)):
		if i >= num_samples_to_visualize:
			break  # Stop after visualizing enough samples

		# Unpack data - assumes test_loader yields (PIL Image, Image Tensor, GT Boxes, GT Labels)
		try:
			pil_img, img_tensor, gt_boxes_raw, gt_labels_raw = test_data
		except ValueError as e:
			print(f"Error unpacking test data batch {i}: {e}. Check test_loader's collate_fn.", file=sys.stderr)
			continue

		# Ensure tensor is on the correct device and add batch dimension
		img_tensor = img_tensor.unsqueeze(0).to(device)
		orig_w, orig_h = pil_img.size  # Get original image dimensions for potential conversions

		# --- Run Inference ---
		with torch.no_grad():
			try:
				# Model forward pass
				# Assumes model returns a list of dicts, take the first element for batch size 1
				predictions = model(img_tensor)
				if isinstance(predictions, list) and len(predictions) > 0:
					preds = predictions[0]
				else:
					# Handle cases where model output might not be a list (shouldn't happen with current wrappers)
					print(f"Warning: Unexpected model output format: {type(predictions)}. Skipping batch {i}.",
						  file=sys.stderr)
					continue

				# Extract predictions (ensure they are on CPU for processing)
				pred_boxes = preds.get("boxes", torch.empty((0, 4))).cpu()
				pred_scores = preds.get("scores", torch.empty((0,))).cpu()
				pred_labels = preds.get("labels", torch.empty((0,))).cpu()

			except Exception as e:
				print(f"Error during model inference or prediction extraction for batch {i}: {e}", file=sys.stderr)
				continue  # Skip this sample

		# --- Filter Predictions ---
		if model_type.lower() == "deformable_detr":
			# No additional filtering needed for DETR
			filtered_boxes = pred_boxes
			filtered_labels = pred_labels
		else:
			keep_indices = pred_scores >= threshold
			filtered_boxes = pred_boxes[keep_indices]
			filtered_labels = pred_labels[keep_indices]

		# --- Process Ground Truth ---
		gt_boxes_absolute = []
		# Ensure raw GT boxes and labels are tensors before processing
		if not isinstance(gt_boxes_raw, torch.Tensor): gt_boxes_raw = torch.tensor(gt_boxes_raw)
		if not isinstance(gt_labels_raw, torch.Tensor): gt_labels_raw = torch.tensor(gt_labels_raw)

		if model_type.lower() == "deformable_detr" and gt_boxes_raw.numel() > 0:
			# For DETR, test dataset provides normalized [cx, cy, w, h]
			# Convert to absolute [x1, y1, x2, y2]
			boxes_cxcywh = gt_boxes_raw.cpu()
			boxes_xyxy = torch.zeros_like(boxes_cxcywh)
			boxes_xyxy[:, 0] = (boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2) * orig_w
			boxes_xyxy[:, 1] = (boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2) * orig_h
			boxes_xyxy[:, 2] = (boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2) * orig_w
			boxes_xyxy[:, 3] = (boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2) * orig_h
			gt_boxes_absolute = boxes_xyxy.tolist()
		elif gt_boxes_raw.numel() > 0:
			# For other models, test dataset provides absolute [x1, y1, x2, y2]
			gt_boxes_absolute = gt_boxes_raw.cpu().tolist()
		# else: gt_boxes_absolute remains []

		# Convert GT labels tensor to list of ints
		gt_labels_list = gt_labels_raw.cpu().tolist()  # labels are expected as 0-indexed from dataset

		# --- Draw Visualizations ---
		try:
			gt_image = pil_img.copy()
			pred_image = pil_img.copy()

			# Draw ground truth in green
			# Always use shift_labels=False for GT drawing as labels are 0-indexed from dataset
			gt_image = draw_boxes(gt_image, gt_boxes_absolute, gt_labels_list, classes, shift_labels=shift_labels,
								  color="green")

			# Draw predictions in red
			# For predictions, always use shift_labels=True for FasterRCNN/RetinaNet (they use 1-indexed labels)
			# and shift_labels=False for DETR (it uses 0-indexed labels)
			pred_image = draw_boxes(pred_image, filtered_boxes, filtered_labels, classes, shift_labels=shift_labels,
									color="red")

			# Combine images side-by-side
			width, height = pil_img.size
			combined = Image.new("RGB", (width * 2, height))
			combined.paste(gt_image, (0, 0))
			combined.paste(pred_image, (width, 0))
			visualizations.append(combined)
		except Exception as e:
			print(f"Error during drawing for batch {i}: {e}", file=sys.stderr)
			continue  # Skip this sample

	# --- Save Combined Visualization ---
	if visualizations:
		total_height = sum(img.height for img in visualizations)
		# Ensure max_width calculation handles case where visualizations list might be empty
		max_width = max(img.width for img in visualizations) if visualizations else 0

		if max_width > 0 and total_height > 0:
			final_img = Image.new("RGB", (max_width, total_height))
			y_offset = 0
			for vis in visualizations:
				final_img.paste(vis, (0, y_offset))
				y_offset += vis.height
			try:
				final_img.save(output_path)
				tqdm.write(f"Saved test visualization to {output_path}")
			except Exception as e:
				print(f"Error saving final visualization image to {output_path}: {e}", file=sys.stderr)
		else:
			print("Warning: Cannot save visualization image due to zero width or height.", file=sys.stderr)
	else:
		tqdm.write("No test images processed or no predictions above threshold; skipping visualization saving.")
