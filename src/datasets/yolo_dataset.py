#!/usr/bin/env python3
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import albumentations as A
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
nested_tensor_from_tensor_list = None
for _path in (
    "torchvision.models.detection._utils",
    "torchvision.ops.misc",
    "torchvision.models.detection.transform",  # <=0.12
):
    try:
        nested_tensor_from_tensor_list = __import__(_path, fromlist=[""]).nested_tensor_from_tensor_list  # type: ignore
        break
    except (ImportError, AttributeError):
        continue
from torchvision.transforms import functional as F

try:
    from google.cloud import storage  # type: ignore
except ImportError:  # pragma: no cover
    storage = None  # streaming from GCS will raise if user requests it


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def load_classes(classes_file: str) -> list[str]:
    """Read class names from a file, one per line."""
    with open(classes_file, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _download_blob(gcs_uri: str, local_path: str) -> None:
    """Download a single object `gs://bucket/obj` to *local_path*."""
    if storage is None:  # pragma: no cover
        raise RuntimeError("google-cloud-storage missing; pip install it or disable stream_gcs.")
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)


def collate_fn(batch):
    """Collate for variable-size inputs (Faster-RCNN / RetinaNet)."""
    return tuple(zip(*batch))


def collate_fn_detr(batch):
    """
    Collate for DETR – prefer the C++ helper when present, else
    fall back to the legacy Python padding loop.
    """
    images, targets = zip(*batch)
    if nested_tensor_from_tensor_list is not None:
        nested = nested_tensor_from_tensor_list(list(images))
        return list(nested.tensors), targets

    # -------- legacy fallback (works on any TorchVision) ----------
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    padded = [
        torch.nn.functional.pad(
            img, (0, max_w - img.shape[2], 0, max_h - img.shape[1])
        )
        for img in images
    ]
    return padded, targets


# --------------------------------------------------------------------------- #
# Main dataset                                                                #
# --------------------------------------------------------------------------- #
class YOLODataset(Dataset):
    """
    Unified dataset that handles training / validation / qualitative test
    visualisation through the *mode* flag.

    Parameters
    ----------
    images_dir : str
        Path or ``gs://`` URI containing ``*.jpg``.
    labels_dir : str
        Path or ``gs://`` URI with YOLO ``*.txt`` files.
    classes_file : str
        Text file with one class name per line.
    mode : \"train\" | \"val\" | \"test\"
        • *train / val*  →  returns ``(tensor, target_dict)`` (TorchVision style)
        • *test*         →  returns ``(pil, tensor, GT_boxes, GT_labels)``
    transforms : Albumentations / TorchVision callable
    normalize_boxes : bool
        Use cxcywh ∈ [0,1] (DETR) or xyxy pixel coordinates (others).
    shift_labels : bool
        +1 offset for models with background class (FR-CNN / Retina).
    stream_gcs : bool
        Lazily download individual files when *images_dir* or *labels_dir*
        starts with ``gs://``.
    """

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
        # ---- Path → str cast ----
        images_dir = str(images_dir)
        labels_dir = str(labels_dir)

        assert mode in {"train", "val", "test"}
        self.mode = mode
        self.transforms = transforms
        self.norm = normalize_boxes
        self.shift = shift_labels
        self.stream_gcs = stream_gcs

        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.classes = load_classes(classes_file)

        if stream_gcs and not images_dir.startswith("gs://"):
            raise ValueError("stream_gcs=True but images_dir is not a gs:// URI.")

        # build list of filenames (no extension)
        if images_dir.startswith("gs://"):
            # query bucket once; store names without fetching files
            bucket_name, prefix = images_dir[5:].split("/", 1)
            if storage is None:  # pragma: no cover
                raise RuntimeError("google-cloud-storage missing.")
            client = storage.Client()
            blobs = client.list_blobs(bucket_name, prefix=prefix)
            self.image_files = sorted(
                [Path(b.name).name for b in blobs if b.name.lower().endswith(".jpg")]
            )
        else:
            self.image_files = sorted(
                f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")
            )

        # local cache dir for streamed objects
        self._cache_dir = Path(tempfile.gettempdir()) / "yolo_stream_cache"
        self._cache_dir.mkdir(exist_ok=True)

    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.image_files)

    # --------------------------------------------------------------------- #
    def _open_image(self, fname: str) -> Image.Image:
        if self.images_dir.startswith("gs://"):
            local_path = self._cache_dir / fname
            if not local_path.exists():
                _download_blob(f"{self.images_dir}/{fname}", str(local_path))
            return Image.open(local_path).convert("RGB")
        return Image.open(os.path.join(self.images_dir, fname)).convert("RGB")

    def _read_label_file(self, stem: str) -> List[Tuple[int, float, float, float, float]]:
        fname = f"{stem}.txt"
        if self.labels_dir.startswith("gs://"):
            local_path = self._cache_dir / fname
            if not local_path.exists():
                _download_blob(f"{self.labels_dir}/{fname}", str(local_path))
            path = local_path
        else:
            path = Path(self.labels_dir) / fname
        if not path.exists():
            return []
        return [tuple(map(float, ln.split())) for ln in open(path)]

    # --------------------------------------------------------------------- #
    def __getitem__(self, idx: int):
        img_name = self.image_files[idx]
        stem = os.path.splitext(img_name)[0]
        pil = self._open_image(img_name)
        W, H = pil.size

        # -------- parse YOLO labels ------------------------------------- #
        boxes, labels = [], []
        for cid, xc, yc, w, h in self._read_label_file(stem):
            cid = int(cid)
            if self.norm:  # DETR branch (cxcywh ∈ [0,1])
                # clip corners to stay valid for Albumentations checks
                x1 = max(0.0, xc - w / 2)
                y1 = max(0.0, yc - h / 2)
                x2 = min(1.0, xc + w / 2)
                y2 = min(1.0, yc + h / 2)
                if x2 <= x1 or y2 <= y1:
                    continue
                xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                boxes.append([xc, yc, w, h])
                labels.append(cid)
            else:  # pixel xyxy branch
                x1 = (xc - w / 2) * W
                y1 = (yc - h / 2) * H
                x2 = (xc + w / 2) * W
                y2 = (yc + h / 2) * H
                x1 = max(0.0, x1)
                y1 = max(0.0, y1)
                x2 = min(W - 1, max(x1, x2))
                y2 = min(H - 1, max(y1, y2))
                boxes.append([x1, y1, x2, y2])
                labels.append(cid + 1 if self.shift else cid)

        # -------- transforms ------------------------------------------- #
        if isinstance(self.transforms, A.Compose):
            out = self.transforms(
                image=np.array(pil), bboxes=boxes, class_labels=labels
            )
            img_tensor = out["image"]
            boxes, labels = list(out["bboxes"]), list(out["class_labels"])
            if self.norm:
                boxes = [
                    [
                        np.clip(xc, 0, 1),
                        np.clip(yc, 0, 1),
                        np.clip(w, 0, 1),
                        np.clip(h, 0, 1),
                    ]
                    for xc, yc, w, h in boxes
                ]
        elif self.transforms:
            img_tensor = self.transforms(pil)
        else:
            img_tensor = F.to_tensor(pil)

        if isinstance(img_tensor, Image.Image):  # pragma: no cover
            img_tensor = F.to_tensor(img_tensor)
        if img_tensor.dtype == torch.uint8:
            img_tensor = img_tensor.float().div_(255)
        if not img_tensor.is_contiguous():
            img_tensor = img_tensor.contiguous()

        # ---------------------------------------------------------------- #
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
                    (boxes_t[:, 3] - boxes_t[:, 1])
                    * (boxes_t[:, 2] - boxes_t[:, 0])
                )
                if boxes_t.numel()
                else torch.tensor([]),
            }
            return img_tensor, target

        # ----------------------------- test/vis ------------------------ #
        return (
            pil,
            img_tensor,
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.int64),
        )
