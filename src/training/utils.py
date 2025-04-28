#!/usr/bin/env python3
from __future__ import annotations

###############################################################################
# Standard-lib / third-party imports
###############################################################################
import csv
import logging
import os
from typing import Any, Dict, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as T
from omegaconf import DictConfig, OmegaConf

###############################################################################
# Logging helper
###############################################################################
DEFAULT_NUM_WORKERS: int = max(2, (os.cpu_count() or 2) // 2)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger (idempotent)."""
    if logging.getLogger().handlers:
        return
    fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    logging.basicConfig(level=level, format=fmt)
    for noisy in ("matplotlib", "PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


###############################################################################
# OmegaConf helpers
###############################################################################
_GROUP_KEYS = {"model", "data", "train", "runtime"}


def flatten_omegaconf(cfg: DictConfig) -> Dict[str, Any]:
    """
    Merge first-level groups (model/, data/, train/, runtime/) into a flat
    :class:`dict` so legacy code that expects ``cfg['batch_size']`` keeps working
    after the Hydra migration.
    """
    plain = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[arg-type]
    merged: Dict[str, Any] = {}
    for k, v in plain.items():
        if k in _GROUP_KEYS and isinstance(v, Mapping):
            merged.update(v)
        else:
            merged[k] = v
    return merged


def get_cfg_dict(cfg: Any) -> Dict[str, Any]:
    """
    Return a plain :class:`dict` regardless of whether *cfg* is a DictConfig or
    an actual mapping (useful for unit tests).
    """
    if isinstance(cfg, DictConfig):
        return flatten_omegaconf(cfg)
    if isinstance(cfg, Mapping):
        return dict(cfg)  # shallow copy
    raise TypeError(
        f"Expected Hydra DictConfig or dict, got object of type {type(cfg).__name__}"
    )


###############################################################################
# Device helpers
###############################################################################
def get_device() -> torch.device:
    """Return the best available :pyclass:`torch.device` in priority order."""
    override = os.getenv("DEVICE")
    if override:
        try:
            dev = torch.device(override)
            logging.getLogger(__name__).info("Using device override: %s", dev)
            return dev
        except Exception:  # pragma: no cover
            logging.getLogger(__name__).warning(
                "Invalid DEVICE override '%s' – ignoring.", override
            )

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def auto_amp_supported(device: torch.device) -> bool:
    """Return ``True`` if *device* supports **native** Automatic Mixed Precision."""
    return device.type in {"cuda", "mps"}


###############################################################################
# Transform helpers (GPU / CPU agnostic)
###############################################################################
def get_simple_transform(
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> T.Compose:
    """Plain ``ToTensor`` + ``Normalize`` transform for val/test."""
    mean = mean or (0.485, 0.456, 0.406)
    std = std or (0.229, 0.224, 0.225)
    return T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])


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
    """Write epoch-level *metrics* (dict of equal-length lists) to a CSV file."""
    keys = list(metrics.keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
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
    """Return a scalar loss from a variety of common detector loss formats."""
    if isinstance(loss_dict, dict):
        return sum(torch.as_tensor(v).sum() for v in loss_dict.values())
    if isinstance(loss_dict, torch.Tensor):
        return loss_dict
    if hasattr(loss_dict, "loss"):
        return torch.as_tensor(loss_dict.loss)
    raise TypeError(
        f"Cannot compute total loss from object of type {type(loss_dict).__name__}"
    )
