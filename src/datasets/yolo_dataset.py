#!/usr/bin/env python3
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from src.transforms import get_pipeline_device

# -------- nested_tensor helper (DETR) ---------------------------------------
nested_tensor_from_tensor_list = None
for _path in (
    "torchvision.models.detection._utils",
    "torchvision.ops.misc",
    "torchvision.models.detection.transform",
):
    try:
        nested_tensor_from_tensor_list = (
            __import__(_path, fromlist=[""]).nested_tensor_from_tensor_list
        )
        break
    except (ImportError, AttributeError):
        continue

# -------- optional GCS streaming -------------------------------------------
try:
    from google.cloud import storage  # type: ignore
except ImportError:  # pragma: no cover
    storage = None  # streaming from GCS will raise if user requests it


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_classes(classes_file: str) -> list[str]:
    if classes_file.startswith("gs://"):
        from tempfile import gettempdir
        cache = Path(gettempdir()) / "grouped_classes.txt"
        if not cache.exists():
            _download_blob(classes_file, str(cache))
        classes_file = cache
    with open(classes_file, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _download_blob(gcs_uri: str, local_path: str) -> None:
    if storage is None:  # pragma: no cover
        raise RuntimeError("google-cloud-storage missing.")
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(local_path)


def collate_fn(batch):        # Faster-RCNN / Retina
    return tuple(zip(*batch))


def collate_fn_detr(batch):   # DETR
    images, targets = zip(*batch)
    if nested_tensor_from_tensor_list is not None:
        nested = nested_tensor_from_tensor_list(list(images))
        return list(nested.tensors), targets
    max_h = max(i.shape[1] for i in images)
    max_w = max(i.shape[2] for i in images)
    return [
        torch.nn.functional.pad(i, (0, max_w - i.shape[2], 0, max_h - i.shape[1]))
        for i in images
    ], targets


# --------------------------------------------------------------------------- #
class YOLODataset(Dataset):
    """Unified dataset for train / val / test splits."""

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        classes_file: str,
        *,
        mode: str = "train",
        transforms=None,
        normalize_boxes: bool = False,
        shift_labels: bool = False,
        stream_gcs: bool = False,
    ):
        images_dir, labels_dir = map(str, (images_dir, labels_dir))
        assert mode in {"train", "val", "test"}
        self.mode, self.transforms = mode, transforms
        self.norm, self.shift, self.stream_gcs = normalize_boxes, shift_labels, stream_gcs
        self.images_dir, self.labels_dir = images_dir, labels_dir
        self.classes = load_classes(classes_file)

        if stream_gcs and not images_dir.startswith("gs://"):
            raise ValueError("stream_gcs=True but images_dir is not a gs:// URI.")

        if images_dir.startswith("gs://"):
            bucket, prefix = images_dir[5:].split("/", 1)
            if storage is None:
                raise RuntimeError("google-cloud-storage missing.")
            blobs = storage.Client().list_blobs(bucket, prefix=prefix)
            self.image_files = sorted(Path(b.name).name for b in blobs if b.name.lower().endswith(".jpg"))
        else:
            self.image_files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(".jpg"))

        self._cache_dir = Path(tempfile.gettempdir()) / "yolo_stream_cache"
        self._cache_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ #
    def __len__(self):  # noqa: D401
        return len(self.image_files)

    # ------------------------------------------------------------------ #
    def _open_image(self, fname: str) -> Image.Image:
        if self.images_dir.startswith("gs://"):
            local = self._cache_dir / fname
            if not local.exists():
                _download_blob(f"{self.images_dir}/{fname}", str(local))
            return Image.open(local).convert("RGB")
        return Image.open(os.path.join(self.images_dir, fname)).convert("RGB")

    def _read_label_file(self, stem: str) -> List[Tuple[int, float, float, float, float]]:
        path = (
            self._cache_dir / f"{stem}.txt"
            if self.labels_dir.startswith("gs://")
            else Path(self.labels_dir) / f"{stem}.txt"
        )
        if self.labels_dir.startswith("gs://") and not path.exists():
            _download_blob(f"{self.labels_dir}/{stem}.txt", str(path))
        if not path.exists():
            return []
        return [tuple(map(float, ln.split())) for ln in open(path)]

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int):
        img_name = self.image_files[idx]
        stem = os.path.splitext(img_name)[0]
        pil = self._open_image(img_name)
        W, H = pil.size

        # -------- YOLO txt → boxes/labels ------------------------------
        boxes, labels = [], []
        for cid, xc, yc, w, h in self._read_label_file(stem):
            cid = int(cid)
            if self.norm:  # cxcywh ∈ [0,1]
                x1, y1 = max(0, xc - w / 2), max(0, yc - h / 2)
                x2, y2 = min(1, xc + w / 2), min(1, yc + h / 2)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
                labels.append(cid)
            else:          # xyxy pixels
                x1, y1 = (xc - w / 2) * W, (yc - h / 2) * H
                x2, y2 = (xc + w / 2) * W, (yc + h / 2) * H
                boxes.append([max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)])
                labels.append(cid + 1 if self.shift else cid)

        # -------- transforms ------------------------------------------
        if self.transforms is not None:
            if isinstance(self.transforms, A.Compose):          # Albumentations CPU
                out = self.transforms(image=np.array(pil), bboxes=boxes, class_labels=labels)
                img_tensor = out["image"]
                boxes, labels = list(out["bboxes"]), list(out["class_labels"])
            elif self.transforms.__class__.__module__.startswith("torchvision.transforms.v2"):
                dev = get_pipeline_device(self.transforms)
                img_t = F.to_tensor(pil).to(dev, non_blocking=True) if dev else F.to_tensor(pil)
                sample = {
                    "image": img_t,
                    "boxes": torch.as_tensor(boxes, dtype=torch.float32, device=dev),
                    "labels": torch.as_tensor(labels, dtype=torch.int64,  device=dev),
                }
                out = self.transforms(sample)
                img_tensor = out["image"]
                boxes = out["boxes"].cpu().tolist() if dev else out["boxes"].tolist()
                labels = out["labels"].cpu().tolist() if dev else out["labels"].tolist()
            else:                                              # Classic TorchVision
                img_tensor = self.transforms(pil)
        else:
            img_tensor = F.to_tensor(pil)

        if img_tensor.dtype == torch.uint8:
            img_tensor = img_tensor.float().div_(255)

        # -------- outputs ---------------------------------------------
        if self.mode in {"train", "val"}:
            boxes_t = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            target = {
                "boxes": boxes_t,
                "labels": labels_t,
                "image_id": torch.tensor([idx]),
                "orig_size": torch.tensor([H, W], dtype=torch.float32),
                "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
                "class_labels": labels_t,
                "area": (
                    (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
                )
                if boxes_t.numel()
                else torch.tensor([]),
            }
            return img_tensor, target

        # -------- qualitative / inference -----------------------------
        return (
            pil,
            img_tensor,
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.int64),
        )
