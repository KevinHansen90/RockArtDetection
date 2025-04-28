#!/usr/bin/env python3

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as F
import albumentations as A


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
    """Return (image_tensor, target_dict) tuples for training/validation."""
    def __init__(
        self,
        images_dir,
        labels_dir,
        classes_file,
        transforms=None,
        normalize_boxes=False      # True for DETR
    ):
        self.images_dir  = images_dir
        self.labels_dir  = labels_dir
        self.transforms  = transforms            # Albumentations or TV2
        self.norm        = normalize_boxes
        self.image_files = sorted(
            f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")
        )
        self.classes     = load_classes(classes_file)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        pil      = Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")
        W, H     = pil.size

        # 1) Read YOLO-format label file
        boxes, labels = [], []
        label_file = os.path.join(
            self.labels_dir, os.path.splitext(img_name)[0] + ".txt"
        )
        if os.path.exists(label_file):
            for line in open(label_file):
                cid, xc, yc, w, h = map(float, line.split())
                cid = int(cid)

                if self.norm:                     # ------------ DETR branch
                    # Clip the *corners* so Albumentations' pre-check passes
                    x1 = xc - w / 2
                    y1 = yc - h / 2
                    x2 = xc + w / 2
                    y2 = yc + h / 2

                    x1 = max(0.0, x1);  y1 = max(0.0, y1)
                    x2 = min(1.0, x2);  y2 = min(1.0, y2)

                    # Skip boxes that are now degenerate or fully outside
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Recompute centre/width/height after clipping
                    xc = (x1 + x2) / 2.0
                    yc = (y1 + y2) / 2.0
                    w  = x2 - x1
                    h  = y2 - y1

                    boxes.append([xc, yc, w, h])
                    labels.append(cid)
                else:                             # ------------ pixel branch
                    x1 = (xc - w / 2) * W
                    y1 = (yc - h / 2) * H
                    x2 = (xc + w / 2) * W
                    y2 = (yc + h / 2) * H
                    # -------- clamp to valid pixel range --------
                    x1 = max(0.0, x1)
                    y1 = max(0.0, y1)
                    x2 = min(W - 1, max(x1, x2))  # ensure x2 ≥ x1
                    y2 = min(H - 1, max(y1, y2))  # ensure y2 ≥ y1
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cid + 1)        # shift for torchvision

        # 2) Apply transforms
        if isinstance(self.transforms, A.Compose):
            out = self.transforms(
                image=np.array(pil), bboxes=boxes, class_labels=labels
            )
            img_tensor = out["image"]
            boxes, labels = list(out["bboxes"]), list(out["class_labels"])

            if self.norm:
                # guarantee numeric safety post-augmentation
                boxes = [
                    [
                        np.clip(xc, 0, 1),
                        np.clip(yc, 0, 1),
                        np.clip(w,  0, 1),
                        np.clip(h,  0, 1),
                    ]
                    for xc, yc, w, h in boxes
                ]
        elif self.transforms:
            img_tensor = self.transforms(pil)
        else:
            img_tensor = F.to_tensor(pil)

        # 3) Ensure tensor dtype/scale
        if isinstance(img_tensor, Image.Image):
            img_tensor = F.to_tensor(img_tensor)
        if img_tensor.dtype == torch.uint8:
            img_tensor = img_tensor.float().div_(255)

        # 4) Pack target dict
        boxes_t  = torch.tensor(boxes,  dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes":      boxes_t,
            "labels":     labels_t,
            "image_id":   torch.tensor([idx]),
            "orig_size":  torch.tensor([H, W], dtype=torch.float32),
            "iscrowd":    torch.zeros(len(labels), dtype=torch.int64),
            "class_labels": labels_t,
            "area": (
                (boxes_t[:, 3] - boxes_t[:, 1]) *
                (boxes_t[:, 2] - boxes_t[:, 0])
            ) if boxes_t.numel() else torch.tensor([]),
        }

        if not img_tensor.is_contiguous():
            img_tensor = img_tensor.contiguous()

        return img_tensor, target


class TestDataset(Dataset):
    """
    Return:
      PIL image, image_tensor, GT_boxes, GT_labels
    Used mainly for qualitative visualisation.
    """
    def __init__(
        self,
        images_dir,
        labels_dir,
        classes_file,
        transforms=None,
        normalize_boxes=False,   # True for DETR
        shift_labels=False       # +1 for Faster-RCNN / RetinaNet
    ):
        self.images_dir  = images_dir
        self.labels_dir  = labels_dir
        self.transforms  = transforms
        self.norm        = normalize_boxes
        self.shift       = shift_labels
        self.image_files = sorted(
            f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")
        )
        self.classes     = load_classes(classes_file)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        pil      = Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")
        W, H     = pil.size

        img_tensor = (
            self.transforms(pil) if self.transforms else F.to_tensor(pil)
        )
        if isinstance(img_tensor, Image.Image):
            img_tensor = F.to_tensor(img_tensor)
        if img_tensor.dtype == torch.uint8:
            img_tensor = img_tensor.float().div_(255)

        boxes, labels = [], []
        label_file = os.path.join(
            self.labels_dir, os.path.splitext(img_name)[0] + ".txt"
        )
        if os.path.exists(label_file):
            for line in open(label_file):
                cid, xc, yc, w, h = map(float, line.split())
                cid = int(cid)

                if self.norm:                     # normalised cxcywh
                    x1 = max(0.0, xc - w / 2)
                    y1 = max(0.0, yc - h / 2)
                    x2 = min(1.0, xc + w / 2)
                    y2 = min(1.0, yc + h / 2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    xc = (x1 + x2) / 2.0
                    yc = (y1 + y2) / 2.0
                    w  = x2 - x1
                    h  = y2 - y1
                    boxes.append([xc, yc, w, h])
                else:                             # absolute xyxy
                    x1 = (xc - w / 2) * W
                    y1 = (yc - h / 2) * H
                    x2 = (xc + w / 2) * W
                    y2 = (yc + h / 2) * H

                    # keep coords inside image bounds
                    x1 = max(0.0, x1)
                    y1 = max(0.0, y1)
                    x2 = min(W - 1, max(x1, x2))
                    y2 = min(H - 1, max(y1, y2))
                    boxes.append([x1, y1, x2, y2])

                labels.append(cid + 1 if self.shift else cid)

        if not img_tensor.is_contiguous():
            img_tensor = img_tensor.contiguous()

        return (
            pil,
            img_tensor,
            torch.tensor(boxes,  dtype=torch.float32),
            torch.tensor(labels, dtype=torch.int64),
        )
