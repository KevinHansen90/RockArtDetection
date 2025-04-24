#!/usr/bin/env python3

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as F


def load_classes(classes_file: str) -> list:
    """Read class names from a file, one per line."""
    with open(classes_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def collate_fn(batch):
    """Collate for variable-size detection targets."""
    return tuple(zip(*batch))


def collate_fn_detr(batch):
    """Collate with padding for DETR: pad all images to the same size."""
    images, targets = zip(*batch)
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    padded_images = []
    for img in images:
        c, h, w = img.shape
        pad_w = max_w - w
        pad_h = max_h - h
        # pad format: (left, right, top, bottom)
        padded = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
        padded_images.append(padded)
    return padded_images, targets


class CustomYOLODataset(Dataset):
    """
    Dataset for training images with YOLO-format labels.
    Returns image tensor and target dict expected by torchvision detectors.
    """
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        classes_file: str,
        transforms=None,
        normalize_boxes: bool = False
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.normalize_boxes = normalize_boxes
        self.image_files = sorted(
            f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')
        )
        self.classes = load_classes(classes_file)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Apply transforms
        if self.transforms:
            img_tensor = self.transforms(image)
        else:
            img_tensor = F.to_tensor(image)

        # Read labels
        label_file = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.labels_dir, label_file)
        boxes, labels = [], []

        if os.path.exists(label_path):
            with open(label_path, "r") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cid = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])

                    # Box coords
                    if self.normalize_boxes:
                        boxes.append([xc, yc, w, h])
                    else:
                        x1 = (xc - w / 2) * orig_w
                        y1 = (yc - h / 2) * orig_h
                        x2 = (xc + w / 2) * orig_w
                        y2 = (yc + h / 2) * orig_h
                        boxes.append([x1, y1, x2, y2])

                    # Shift for Faster R-CNN & RetinaNet
                    if self.normalize_boxes:
                        labels.append(cid)
                    else:
                        labels.append(cid + 1)

        # Format tensors
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.float32)
        }
        if boxes_tensor.numel():
            target["area"] = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0])
        else:
            target["area"] = torch.tensor([])
        target["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)
        target["class_labels"] = target["labels"]

        return img_tensor, target


class TestDataset(Dataset):
    """
    Test dataset for inference & visualization.
    Returns:
      - PIL Image for drawing
      - Tensor for model input
      - boxes and labels aligned to model output indexing
    """
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        classes_file: str,
        transforms=None,
        normalize_boxes: bool = False,
        shift_labels: bool = False
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.normalize_boxes = normalize_boxes
        self.shift_labels = shift_labels
        self.image_files = sorted(
            f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')
        )
        self.classes = load_classes(classes_file)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        pil_img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = pil_img.size

        # Transform for model
        img_copy = pil_img.copy()
        if self.transforms:
            img_tensor = self.transforms(img_copy)
        else:
            img_tensor = F.to_tensor(img_copy)

        # Parse YOLO labels
        label_file = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.labels_dir, label_file)
        boxes, labels = [], []

        if os.path.exists(label_path):
            with open(label_path, "r") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cid = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])

                    # Box coords
                    if self.normalize_boxes:
                        boxes.append([xc, yc, w, h])
                    else:
                        x1 = (xc - w / 2) * orig_w
                        y1 = (yc - h / 2) * orig_h
                        x2 = (xc + w / 2) * orig_w
                        y2 = (yc + h / 2) * orig_h
                        boxes.append([x1, y1, x2, y2])

                    # Shift only if flag set (Faster R-CNN uses +1)
                    label_val = cid + 1 if self.shift_labels else cid
                    labels.append(label_val)

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        # Return PIL image for drawing, tensor for inference, and GT data
        return pil_img, img_tensor, boxes_tensor, labels_tensor

