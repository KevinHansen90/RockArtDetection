import os
import sys
import torch
import argparse
import yaml
from torchvision.utils import draw_bounding_boxes
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import torchvision.ops as ops
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np

from utils.utils import load_model, collate_fn, collate_fn_detr
from data.dataset import RockArtDataset
from data.transforms import get_transforms
from evaluation.validation import eval_forward

import subprocess

# Define a list of colors for different classes
COLORS = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta", "lime", "pink"]


def load_yaml_config(config_path):
	with open(config_path, 'r') as file:
		config = yaml.safe_load(file)
	return config


def visualize_boxes(image, boxes, labels, scores=None, score_threshold=0.8, label_prefix=""):
	boxes = boxes.int()
	labels = labels.int()

	if scores is not None:
		# Filter out boxes below the score threshold
		valid_indices = scores > score_threshold
		boxes = boxes[valid_indices]
		labels = labels[valid_indices]
		scores = scores[valid_indices]
		label_strings = [f"{label_prefix}{label.item()} {score:.2f}" for label, score in zip(labels, scores)]
	else:
		label_strings = [f"{label_prefix}{label.item()}" for label in labels]

	colors = [COLORS[label.item() % len(COLORS)] for label in labels]

	# Convert image to uint8
	if image.dtype != torch.uint8:
		image = (image * 255).to(torch.uint8).cpu()

	image_with_boxes = draw_bounding_boxes(image, boxes, labels=label_strings, colors=colors, width=2)
	return image_with_boxes


def combine_images_side_by_side(img1_pil, img2_pil):
	total_width = img1_pil.width + img2_pil.width
	max_height = max(img1_pil.height, img2_pil.height)
	combined_image = Image.new('RGB', (total_width, max_height))
	combined_image.paste(img1_pil, (0, 0))
	combined_image.paste(img2_pil, (img1_pil.width, 0))
	return combined_image


def plot_comparison(image_tensor, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, image_title):
	# Visualize ground truth
	img_with_gt = visualize_boxes(image_tensor.clone(), gt_boxes, gt_labels, label_prefix="GT")
	img_with_gt_pil = F.to_pil_image(img_with_gt)

	# Visualize predictions
	if len(pred_boxes) > 0:
		img_with_pred = visualize_boxes(image_tensor.clone(), pred_boxes, pred_labels, pred_scores, label_prefix="P")
	else:
		img_with_pred = image_tensor.clone()
	img_with_pred_pil = F.to_pil_image(img_with_pred)

	# Combine images side by side
	combined_image = combine_images_side_by_side(img_with_gt_pil, img_with_pred_pil)

	# Add image title
	draw = ImageDraw.Draw(combined_image)
	font = ImageFont.load_default()
	draw.text((10, 10), image_title, fill="white", font=font)

	return combined_image  # Return the combined PIL image


def assemble_comparison_image(image_list):
	# Determine the width and height of the final image
	widths, heights = zip(*(img.size for img in image_list))

	max_width = max(widths)
	total_height = sum(heights)

	final_image = Image.new('RGB', (max_width, total_height), color='white')

	y_offset = 0

	for img in image_list:
		final_image.paste(img, (0, y_offset))
		y_offset += img.height

	return final_image


def inference_yolo(model_path, data_path, output_dir):
	# Ensure output directory exists
	os.makedirs(output_dir, exist_ok=True)

	# Define command to run detect.py
	command = [
		'python', '/Users/kevinhansen/Documents/Git/external_repos/yolov5/detect.py',
		'--weights', model_path,
		'--source', os.path.join(data_path, 'images'),
		'--img', '640',
		'--conf', '0.25',
		'--iou', '0.45',
		'--save-txt',
		'--save-conf',
		'--project', output_dir,
		'--name', 'inference_results',
		'--exist-ok'
	]

	print("Running YOLOv5 inference...")
	subprocess.run(command)
	print("YOLOv5 inference completed.")


def inference(model, device, data_loader, output_dir, model_type):
	model.eval()
	os.makedirs(output_dir, exist_ok=True)
	comparison_images = []

	dataset = data_loader.dataset
	sample_idx = 0  # Initialize sample index

	for batch_idx, batch in enumerate(data_loader):
		print(f"Processing batch {batch_idx}")
		if model_type == 'deformable_detr':
			pixel_values = batch["pixel_values"].to(device)
			targets = batch["labels"]

			with torch.no_grad():
				outputs = model(pixel_values=pixel_values)

			# Extract predictions
			pred_boxes = outputs.pred_boxes
			pred_logits = outputs.logits

			for i in range(len(pixel_values)):
				# Get image path from dataset
				image_path = dataset.imgs[sample_idx]
				img_pil = Image.open(os.path.join(
					dataset.root, "images", image_path)).convert("RGB")
				original_size = img_pil.size
				img_pil = img_pil.resize(dataset.image_size)
				image_tensor = F.to_tensor(img_pil)

				# Get predictions for the image
				boxes = pred_boxes[i]
				logits = pred_logits[i]

				# Convert boxes from cxcywh to xyxy and scale to image size
				boxes = ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
				boxes[:, [0, 2]] *= dataset.image_size[0]  # Width scaling
				boxes[:, [1, 3]] *= dataset.image_size[1]  # Height scaling

				# Get scores and labels
				probs = logits.softmax(-1)
				scores, labels_pred = probs[..., :-1].max(-1)

				# Get ground truth boxes and labels
				gt = targets[i]

				# Extract boxes and labels from annotations
				gt_boxes = gt['boxes']
				gt_labels = gt['class_labels']

				# Ensure gt_boxes are tensors
				gt_boxes = gt_boxes.float()
				gt_labels = gt_labels.long()

				# Convert boxes from YOLO format (x_center, y_center, width, height) to (xmin, ymin, xmax, ymax)
				gt_boxes_xyxy = torch.zeros_like(gt_boxes)
				gt_boxes_xyxy[:, 0] = gt_boxes[:, 0] - gt_boxes[:, 2] / 2  # xmin
				gt_boxes_xyxy[:, 1] = gt_boxes[:, 1] - gt_boxes[:, 3] / 2  # ymin
				gt_boxes_xyxy[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2] / 2  # xmax
				gt_boxes_xyxy[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3] / 2  # ymax

				# Scale ground truth boxes to the resized image size
				scale_x = dataset.image_size[0]
				scale_y = dataset.image_size[1]
				gt_boxes_xyxy[:, [0, 2]] *= scale_x
				gt_boxes_xyxy[:, [1, 3]] *= scale_y

				# Use gt_boxes_xyxy instead of gt_boxes in the plot_comparison function
				combined_image_pil = plot_comparison(
					image_tensor, boxes.cpu(), labels_pred.cpu(), scores.cpu(),
					gt_boxes_xyxy.cpu(), gt_labels.cpu(), f"Image {sample_idx}"
				)
				comparison_images.append(combined_image_pil)
				sample_idx += 1  # Increment sample index

		elif model_type in ['retinanet', 'faster_rcnn']:
			images = batch[0]
			targets = batch[1]

			images = list(img.to(device) for img in images)
			with torch.no_grad():
				predictions = model(images)

			for i in range(len(images)):
				# Get image path from dataset
				image_path = dataset.imgs[sample_idx]
				img_pil = Image.open(os.path.join(dataset.root, "images", image_path)).convert('RGB')
				img_pil = img_pil.resize(dataset.image_size)
				image_tensor = F.to_tensor(img_pil)

				pred = predictions[i]
				boxes = pred['boxes']
				labels_pred = pred['labels']
				scores = pred['scores']

				gt = targets[i]
				gt_boxes = gt['boxes']
				gt_labels = gt['labels']

				# Plot comparison and collect images
				combined_image_pil = plot_comparison(
					image_tensor, boxes.cpu(), labels_pred.cpu(), scores.cpu(),
					gt_boxes.cpu(), gt_labels.cpu(), f"Image {sample_idx}"
				)
				comparison_images.append(combined_image_pil)
				sample_idx += 1  # Increment sample index
		else:
			print(f"Model type '{model_type}' not recognized.")
			continue

		print(f"Processed batch {batch_idx}")

	# After processing all batches, create a single image
	if comparison_images:
		final_image = assemble_comparison_image(comparison_images)
		final_image.save(os.path.join(output_dir, 'comparison_all_images.png'))
		print(f"Saved combined comparison image to {output_dir}")
	else:
		print("No comparison images were generated.")


def main(config_path, model_path, output_dir):
	# Load configuration
	config = load_yaml_config(config_path)
	device = torch.device(config['device'])
	model_type = config['model']['type']

	# Process checkpoint_dir to include train_transforms if needed
	train_transforms = config.get('train_transforms', 'none')
	checkpoint_dir = config['checkpoint_dir'].replace('${train_transforms}', train_transforms)
	config['checkpoint_dir'] = checkpoint_dir

	# If --model is not provided, use the model path from checkpoint_dir
	if not model_path:
		default_model_file = 'last.pth' if model_type != 'yolo' else 'last.pt'
		model_path = os.path.join(checkpoint_dir, default_model_file)

	# If --output_dir is not provided, use checkpoint_dir as output_dir
	if not output_dir:
		output_dir = checkpoint_dir

	if model_type == 'yolo':
		# Run inference using detect.py
		data_path = config['test_dir']
		inference_yolo(model_path, data_path, output_dir)
	else:
		# Load model
		model = load_model(config['model'])
		state_dict = torch.load(model_path, map_location=device)
		model.load_state_dict(state_dict, strict=False)
		model.to(device)

		# Load dataset
		test_transform = get_transforms(
			config['val_transforms'],
			train=False,
			image_size=tuple(config.get('image_size', [512, 512]))
		) if config['val_transforms'] else None

		test_dataset = RockArtDataset(
			config['test_dir'],
			transforms=test_transform,
			image_size=tuple(config.get('image_size', [512, 512])),
			model_type=model_type
		)

		# Create data loader
		if model_type == 'deformable_detr':
			test_loader = DataLoader(test_dataset, collate_fn=collate_fn_detr, **config['val_dataloader'])
		else:
			test_loader = DataLoader(test_dataset, collate_fn=collate_fn, **config['val_dataloader'])

		# Perform inference
		inference(model, device, test_loader, output_dir, model_type)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Rock Art Detection Inference')
	parser.add_argument('--config', required=True, help='Path to the config file')
	parser.add_argument('--model', help='Path to the model file')
	parser.add_argument('--output_dir', help='Directory to save the results')

	args = parser.parse_args()

	# Load the YAML config
	config = load_yaml_config(args.config)

	main(args.config, args.model, args.output_dir)

# python inference.py --config configs/config_faster_rcnn.yaml
