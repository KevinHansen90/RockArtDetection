#!/usr/bin/env python3
from __future__ import annotations

import csv
import logging
import os
from typing import Any, Mapping, Sequence, Tuple

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as T
import yaml
from albumentations.pytorch import ToTensorV2

###############################################################################
# Configuration & logging helpers
###############################################################################

#: Default number of worker processes for ``torch.utils.data.DataLoader`` when
#: the YAML config omits *num_workers*.
DEFAULT_NUM_WORKERS: int = max(2, (os.cpu_count() or 2) // 2)


def setup_logging(level: int = logging.INFO) -> None:
    """Initialise the root *and* library loggers.

    * Call this **once** at program start (e.g. in ``train.py``) **before** any
      sub‑modules log.
    * Guard against double‑initialisation when unit‑tests import this module
      repeatedly.
    """
    root = logging.getLogger()
    if root.handlers:  # already configured in this process – do nothing
        return

    fmt = "%(asctime)s [%(levelname)s] %(name)s ‑ %(message)s"
    logging.basicConfig(level=level, format=fmt)  # adds a *StreamHandler*

    # Silence noisy third‑party libraries at INFO by default
    for noisy in ("matplotlib", "PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


###############################################################################
# Device helpers
###############################################################################

def get_device() -> torch.device:
    """Return the preferred :pyclass:`torch.device` following this order:

    1. Environment variable ``DEVICE`` (if valid),
    2. CUDA,
    3. Apple‑Metal (MPS),
    4. CPU.
    """
    override = os.getenv("DEVICE")
    if override:
        try:
            dev = torch.device(override)
            logging.getLogger(__name__).info("Using device override: %s", dev)
            return dev
        except Exception:
            logging.getLogger(__name__).warning(
                "Invalid DEVICE override '%s' – ignoring.", override
            )

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def auto_amp_supported(device: torch.device) -> bool:
    """Return ``True`` if *device* supports **native* Automatic Mixed Precision.

    CUDA (sm >= 7.x) and Apple‑Silicon (MPS) both do; CPU currently does not.
    """
    return device.type in {"cuda", "mps"}


###############################################################################
# YAML + config helpers
###############################################################################

def load_config(path: str) -> dict[str, Any]:
    """Load a YAML file and return it as a dictionary."""
    with open(path, "r", encoding="utf‑8") as f:
        cfg = yaml.safe_load(f)
    logging.getLogger(__name__).info("Loaded config %s", path)
    return cfg


###############################################################################
# Transform builders
###############################################################################

def get_simple_transform(
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> T.Compose:
    """Plain ToTensor + Normalize transform."""
    mean = mean or (0.485, 0.456, 0.406)
    std = std or (0.229, 0.224, 0.225)
    return T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])


_BASE_AUGS: list[A.BasicTransform] = [
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(0.1, 0.1, p=0.3),
    A.CLAHE(clip_limit=2.0, p=0.2),
    A.LongestMaxSize(max_size=1024, p=1.0),
]


def get_train_transform(
    is_detr: bool,
    mps_safe: bool,
    seed: int | None = None,
) -> A.Compose:
    """Build one Albumentations pipeline shared by all detectors."""
    ops: list[A.BasicTransform] = list(_BASE_AUGS)  # shallow‑copy

    if not mps_safe:  # add heavier geometric augments on CUDA/CPU
        ops.insert(1, A.Affine(translate_percent=0.1, scale=(0.85, 1.15), p=0.5))
        ops.insert(2, A.RandomRotate90(p=0.25))

    # Always ensure H and W are divisible by 32 for detector backbones
    ops.append(
        A.PadIfNeeded(
            min_height=None,
            min_width=None,
            pad_height_divisor=32,
            pad_width_divisor=32,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
        )
    )
    ops.append(ToTensorV2())

    fmt = "yolo" if is_detr else "pascal_voc"
    return A.Compose(
        ops,
        bbox_params=A.BboxParams(
            format=fmt,
            label_fields=["class_labels"],
            min_visibility=0.4,
            check_each_transform=False,
        ),
        seed=seed,
    )


###############################################################################
# Plotting + CSV helpers
###############################################################################

def plot_curve(
    values: Sequence[float],
    ylabel: str,
    title: str,
    output_path: str,
    figsize: Tuple[int, int] = (8, 6),
    marker: str = "o",
) -> None:
    """Plot *values* vs epoch and save the figure to *output_path*."""
    try:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(range(1, len(values) + 1), values, marker=marker)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        logging.getLogger(__name__).debug("Saved plot '%s' → %s", title, output_path)
    except Exception as exc:  # pragma: no cover
        logging.getLogger(__name__).error("Failed to plot '%s': %s", title, exc)


def save_metrics_csv(csv_path: str, metrics: Mapping[str, Sequence[Any]]) -> None:
    """Write epoch‑level *metrics* (dict of equal‑length lists) to CSV."""
    keys = list(metrics.keys())
    with open(csv_path, "w", newline="", encoding="utf‑8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(zip(*(metrics[k] for k in keys)))
    logging.getLogger(__name__).debug("Saved metrics CSV → %s", csv_path)


###############################################################################
# Model helpers
###############################################################################

def freeze_batchnorm(model: nn.Module, backbone_only: bool = True) -> nn.Module:
    """Set all ``BatchNorm2d`` layers to eval mode and disable their grads."""
    mods = (
        model.backbone.modules()
        if backbone_only and hasattr(model, "backbone")
        else model.modules()
    )
    frozen = 0
    for m in mods:
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            frozen += 1
    logging.getLogger(__name__).info("Frozen %d BatchNorm2d layers.", frozen)
    return model


###############################################################################
# Loss helpers
###############################################################################

def compute_total_loss(loss_dict: Any) -> torch.Tensor:
    """Return a scalar loss from various possible *loss_dict* formats.

    * ``dict`` of tensors – sums the (already scalar) tensors.
    * bare ``torch.Tensor`` – returned untouched.
    * HF ``ModelOutput`` – uses its ``loss`` attribute.
    """
    if isinstance(loss_dict, dict):
        return sum(torch.as_tensor(v).sum() for v in loss_dict.values())
    if isinstance(loss_dict, torch.Tensor):
        return loss_dict
    if hasattr(loss_dict, "loss"):
        return torch.as_tensor(loss_dict.loss)
    raise TypeError(
        f"Cannot compute total loss from object of type {type(loss_dict).__name__}"
    )
