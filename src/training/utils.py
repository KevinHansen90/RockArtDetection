#!/usr/bin/env python3

import os
import csv
import yaml
import logging
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as T

# Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_config(path: str) -> dict:
    """Read and parse a YAML config file, raising on errors."""
    try:
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        logger.info(f"Loaded config from {path}")
        return cfg
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


def get_device() -> torch.device:
    """
    Determine compute device:
    - Honor DEVICE env var if set (e.g., cuda:0, cpu, mps)
    - Otherwise prefer CUDA, then MPS, then CPU
    """
    override = os.getenv('DEVICE')
    if override:
        try:
            device = torch.device(override)
            logger.info(f"Using device override: {device}")
            return device
        except Exception:
            logger.warning(f"Invalid DEVICE override '{override}', falling back.")
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_simple_transform(mean=None, std=None) -> T.Compose:
    """
    Return a basic transform: ToTensor + Normalize.
    Defaults to ImageNet stats if mean/std not provided.
    """
    mean = mean or [0.485, 0.456, 0.406]
    std  = std  or [0.229, 0.224, 0.225]
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])


def plot_curve(
    values: list,
    ylabel: str,
    title: str,
    output_path: str,
    figsize: tuple = (8, 6),
    marker: str = 'o'
) -> None:
    """
    Plot a sequence of values and save to file.
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(range(1, len(values)+1), values, marker=marker)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        fig.savefig(output_path)
        plt.close(fig)
        logger.info(f"Saved plot '{title}' to {output_path}")
    except Exception as e:
        logger.error(f"Failed to plot '{title}': {e}")


def save_metrics_csv(csv_path: str, metrics: dict):
    """
    Save epoch‐level metrics (all keys in metrics dict) to CSV.
    metrics is a dict of lists, all of same length.
    """
    # Header is the dict keys
    keys = list(metrics.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(len(metrics[keys[0]])):
            row = [metrics[k][i] for k in keys]
            writer.writerow(row)
    from tqdm import tqdm; tqdm.write(f"Saved metrics CSV to {csv_path}")


def freeze_batchnorm(model: nn.Module, backbone_only: bool = True) -> nn.Module:
    """
    Freeze all BatchNorm2d layers.
    If backbone_only, only freeze layers under model.backbone.
    """
    modules = model.backbone.modules() if backbone_only and hasattr(model, 'backbone') else model.modules()
    frozen = 0
    for m in modules:
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            frozen += 1
    logger.info(f"Frozen {frozen} BatchNorm2d layers.")
    return model


def compute_total_loss(loss_dict) -> torch.Tensor:
    """
    Sum losses when model returns a dict of tensors.
    If a single Tensor is returned, return it directly.
    """
    if isinstance(loss_dict, dict):
        total = torch.tensor(0.0, device=next(iter(loss_dict.values())).device)
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                total = total + v.sum()
            else:
                total = total + torch.tensor(v)
        return total
    if isinstance(loss_dict, torch.Tensor):
        return loss_dict
    # Fallback: try attribute 'loss'
    try:
        return loss_dict.loss
    except Exception:
        raise ValueError(f"Cannot compute total loss from object: {loss_dict}")