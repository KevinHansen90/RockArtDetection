# src/datasets/yolo_dataset.py

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as F


def load_classes(classes_file):
	with open(classes_file, "r") as f:
		return [line.strip() for line in f.readlines()]


def collate_fn(batch):
	"""Custom collate function for variable-size detection targets."""
	return tuple(zip(*batch))


def collate_fn_detr(batch):
	images, targets = zip(*batch)
	# Determine the maximum height and width in this batch.
	max_h = max(img.shape[1] for img in images)
	max_w = max(img.shape[2] for img in images)
	padded_images = []
	for img in images:
		c, h, w = img.shape
		# Compute the padding amounts (pad right and bottom)
		pad_w = max_w - w
		pad_h = max_h - h
		# Pad format is (left, right, top, bottom)
		padded_img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
		padded_images.append(padded_img)
	return padded_images, targets


class CustomYOLODataset(Dataset):
	"""Dataset for training images with YOLO-format labels."""

	def __init__(self, images_dir, labels_dir, classes_file, transforms=None, normalize_boxes=False):
		"""
		Args:
			images_dir (str): Directory with images.
			labels_dir (str): Directory with YOLO-format labels.
			classes_file (str): Path to file listing class names.
			transforms: Optional transform to be applied on an image.
			normalize_boxes (bool): If True, return bounding boxes in normalized coordinates [0,1];
									if False, return absolute pixel coordinates.
		"""
		self.images_dir = images_dir
		self.labels_dir = labels_dir
		self.transforms = transforms
		self.normalize_boxes = normalize_boxes
		self.image_files = sorted([
			f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")
		])
		self.classes = load_classes(classes_file)

	def __len__(self):
		return len(self.image_files)

	def __getitem__(self, idx):
		img_filename = self.image_files[idx]
		img_path = os.path.join(self.images_dir, img_filename)
		image = Image.open(img_path).convert("RGB")

		if self.transforms:
			image = self.transforms(image)
		else:
			image = F.to_tensor(image)

		# Build target from YOLO label
		label_filename = os.path.splitext(img_filename)[0] + ".txt"
		label_path = os.path.join(self.labels_dir, label_filename)
		boxes, labels = [], []

		# Get original image dimensions
		with Image.open(img_path) as img:
			orig_w, orig_h = img.size  # <-- NEW: Get original dimensions

		if os.path.exists(label_path):
			with open(label_path, "r") as f:
				lines = f.readlines()
			for line in lines:
				if not line.strip():
					continue
				parts = line.strip().split()
				class_id = int(parts[0])
				x_center, y_center, width, height = map(float, parts[1:])

				if self.normalize_boxes:
					# Directly use YOLO's normalized center coordinates <-- FIXED
					boxes.append([x_center, y_center, width, height])
				else:
					# Convert to absolute coordinates
					x1 = (x_center - width / 2) * orig_w
					y1 = (y_center - height / 2) * orig_h
					x2 = (x_center + width / 2) * orig_w
					y2 = (y_center + height / 2) * orig_h
					boxes.append([x1, y1, x2, y2])

				if self.normalize_boxes:
					# For deformable_detr, do not shift the label.
					labels.append(class_id)
				else:
					# For Faster R-CNN and RetinaNet, shift the label by +1.
					labels.append(class_id + 1)

		boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
		labels = torch.as_tensor(labels, dtype=torch.int64)
		target = {
			"boxes": boxes,
			"labels": labels,
			"image_id": torch.tensor([idx], dtype=torch.int64),
			"orig_size": torch.tensor([orig_h, orig_w], dtype=torch.float32)  # <-- NEW
		}

		if boxes.numel() > 0:
			target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
		else:
			target["area"] = torch.tensor([])

		target["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)
		target["class_labels"] = target["labels"]
		return image, target


class TestDataset(Dataset):
	"""
	Test dataset that returns the original PIL image, transformed tensor,
	ground truth boxes, and labels.
	"""

	def __init__(self, images_dir, labels_dir, classes_file, transforms=None, normalize_boxes=False):
		"""
		Args:
			images_dir (str): Directory with images.
			labels_dir (str): Directory with YOLO-format labels.
			classes_file (str): Path to file listing class names.
			transforms: Optional transform to be applied on an image.
			normalize_boxes (bool): If True, return bounding boxes in normalized coordinates [0,1];
									if False, return absolute pixel coordinates.
		"""
		self.images_dir = images_dir
		self.labels_dir = labels_dir
		self.transforms = transforms
		self.normalize_boxes = normalize_boxes
		self.image_files = sorted(
			f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")
		)
		self.classes = load_classes(classes_file)

	def __len__(self):
		return len(self.image_files)

	def __getitem__(self, idx):
		img_filename = self.image_files[idx]
		img_path = os.path.join(self.images_dir, img_filename)
		pil_image = Image.open(img_path).convert("RGB")
		orig_w, orig_h = pil_image.size  # <-- NEW: Get original dimensions

		# Create a copy for transforms
		transformed_image = pil_image.copy()
		if self.transforms:
			image_tensor = self.transforms(transformed_image)
		else:
			image_tensor = F.to_tensor(transformed_image)

		# Build ground-truth
		label_filename = os.path.splitext(img_filename)[0] + ".txt"
		label_path = os.path.join(self.labels_dir, label_filename)
		boxes, labels = [], []

		if os.path.exists(label_path):
			with open(label_path, "r") as f:
				lines = f.readlines()
			for line in lines:
				parts = line.strip().split()
				if not parts:
					continue
				class_id = int(parts[0])
				num_classes = len(self.classes)
				assert 0 <= class_id < num_classes, \
					f"Invalid class_id {class_id} found in {label_path}. Expected range [0, {num_classes - 1}]"
				x_center, y_center, width, height = map(float, parts[1:])

				if self.normalize_boxes:
					# Store normalized center coordinates <-- FIXED
					boxes.append([x_center, y_center, width, height])
				else:
					# Convert to absolute coordinates
					x1 = (x_center - width / 2) * orig_w
					y1 = (y_center - height / 2) * orig_h
					x2 = (x_center + width / 2) * orig_w
					y2 = (y_center + height / 2) * orig_h
					boxes.append([x1, y1, x2, y2])

				if self.normalize_boxes:
					# For deformable_detr, do not shift the label.
					labels.append(class_id)
				else:
					# For Faster R-CNN and RetinaNet, shift the label by +1.
					labels.append(class_id + 1)

		boxes = torch.as_tensor(boxes, dtype=torch.float32)
		labels = torch.as_tensor(labels, dtype=torch.int64)

		# Return original dimensions for visualization
		return (pil_image, image_tensor, boxes, labels)
