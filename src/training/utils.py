import csv
import logging
import os
from functools import lru_cache
from typing import Sequence, Tuple, Optional

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.v2 as TV
import yaml
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_config(path: str) -> dict:
    """Load a YAML file and return it as a dictionary."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info("Loaded config %s", path)
    return cfg


def get_device() -> torch.device:
    """
    Return the preferred `torch.device` following this order:

    1. Honour `DEVICE` environment variable (if valid),
    2. CUDA,
    3. Apple-Metal (MPS),
    4. CPU.
    """
    override = os.getenv("DEVICE")
    if override:
        try:
            dev = torch.device(override)
            logger.info("Using device override: %s", dev)
            return dev
        except Exception:
            logger.warning("Invalid DEVICE override '%s' – ignoring.", override)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_simple_transform(
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> T.Compose:
    """Plain ToTensor + Normalize transform."""
    mean = mean or (0.485, 0.456, 0.406)
    std = std or (0.229, 0.224, 0.225)
    return T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])


_BASE_AUGS = [
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
    """
    Build one Albumentations pipeline shared by all detectors.

    Parameters
    ----------
    is_detr  : True for Deformable-DETR (YOLO boxes), False for FR-CNN/Retina.
    mps_safe : Skip affine + rotate on Apple-Metal to avoid shader crashes.
    seed     : Optional deterministic seed for reproducibility.

    Returns
    -------
    albumentations.Compose
    """
    ops = list(_BASE_AUGS)  # copy to avoid global mutation

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


def plot_curve(
    values: Sequence[float],
    ylabel: str,
    title: str,
    output_path: str,
    figsize: Tuple[int, int] = (8, 6),
    marker: str = "o",
) -> None:
    """Plot a 1-D curve and save it to *output_path*."""
    try:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(range(1, len(values) + 1), values, marker=marker)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        logger.debug("Saved plot '%s' → %s", title, output_path)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to plot '%s': %s", title, exc)


def save_metrics_csv(csv_path: str, metrics: dict) -> None:
    """Write epoch-level metrics (dict of equal-length lists) to CSV."""
    keys = list(metrics.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(zip(*(metrics[k] for k in keys)))
    logger.debug("Saved metrics CSV → %s", csv_path)


def freeze_batchnorm(model: nn.Module, backbone_only: bool = True) -> nn.Module:
    """Set BatchNorm2d layers to eval + disable grads."""
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
    logger.info("Frozen %d BatchNorm2d layers.", frozen)
    return model


def compute_total_loss(loss_dict) -> torch.Tensor:
    """
    Sum a dict of loss tensors or return the tensor directly.
    """
    if isinstance(loss_dict, dict):
        total = torch.zeros((), device=next(iter(loss_dict.values())).device)
        for v in loss_dict.values():
            total += v.sum() if isinstance(v, torch.Tensor) else torch.as_tensor(v)
        return total
    if isinstance(loss_dict, torch.Tensor):
        return loss_dict
    if hasattr(loss_dict, "loss"):  # for HF models
        return loss_dict.loss
    raise ValueError(f"Cannot compute total loss from object of type {type(loss_dict)}")
