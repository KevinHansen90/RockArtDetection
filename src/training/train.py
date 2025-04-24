#!/usr/bin/env python3
import os
# Force MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"]   = ":4096:8"

import argparse
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import random
import shutil
import logging
import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.utils import (
    load_config, get_device, get_simple_transform,
    plot_curve, save_metrics_csv
)
from src.training.engine import train_model
from src.models.detection_models import (
    get_detection_model, get_optimizer, get_scheduler
)
from src.datasets.yolo_dataset import (
    CustomYOLODataset, TestDataset, collate_fn, collate_fn_detr, load_classes
)
from src.training.evaluate import evaluate_and_visualize

# Logger setup
logger = logging.getLogger("train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Train an object detection model")
    parser.add_argument("--config",     required=True, help="Path to YAML config")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    opts = parser.parse_args()

    # Load config + set seeds
    cfg = load_config(opts.config)
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Prepare experiment dirs
    exp_dir    = os.path.join("experiments", opts.experiment)
    ckpt_dir   = os.path.join(exp_dir, "checkpoints")
    log_dir    = os.path.join(exp_dir, "logs")
    result_dir = os.path.join(exp_dir, "results")
    for d in (exp_dir, ckpt_dir, log_dir, result_dir):
        os.makedirs(d, exist_ok=True)
    shutil.copy2(opts.config, os.path.join(exp_dir, "used_config.yaml"))
    logger.info(f"Experiment directory: {exp_dir}")

    # Device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Data paths
    classes      = load_classes(cfg["classes_file"])
    data_root    = cfg["data_root"]
    train_imgs   = os.path.join(data_root, "train/images")
    train_lbls   = os.path.join(data_root, "train/labels")
    val_imgs     = os.path.join(data_root, "val/images")
    val_lbls     = os.path.join(data_root, "val/labels")
    test_imgs    = os.path.join(data_root, "test/images")
    test_lbls    = os.path.join(data_root, "test/labels")

    # Dataset & Dataloader
    is_detr = (cfg["model_type"].lower() == "deformable_detr")
    coll     = collate_fn_detr if is_detr else collate_fn
    transform = get_simple_transform()

    train_ds = CustomYOLODataset(train_imgs, train_lbls, cfg["classes_file"], transform, is_detr)
    val_ds   = CustomYOLODataset(val_imgs,   val_lbls, cfg["classes_file"], transform, is_detr)

    g = torch.Generator().manual_seed(seed)
    dl_kwargs = {
        "batch_size":    cfg.get("batch_size", 2),
        "shuffle":       True,
        "num_workers":   cfg.get("num_workers", 0),
        "pin_memory":    True,
        "collate_fn":    coll,
        "generator":     g,
    }
    if cfg.get("num_workers", 0) > 0:
        dl_kwargs.update({"prefetch_factor": 2, "persistent_workers": True})

    train_loader = DataLoader(train_ds, **dl_kwargs)
    val_loader   = DataLoader(
        val_ds,
        **{**dl_kwargs, "shuffle": False}
    )

    # Model, optimizer, scheduler
    model_type = cfg["model_type"].lower()
    num_classes = len(classes) + (0 if is_detr else 1)
    model   = get_detection_model(model_type, num_classes, config=cfg).to(device)
    optim   = get_optimizer(model, cfg)
    sched   = get_scheduler(optim, cfg)

    # Pass log_dir for TensorBoard inside engine.py
    cfg["log_dir"] = log_dir

    # Train & collect metrics
    metrics_out = train_model(model, train_loader, val_loader, device, optim, sched, cfg)
    (
        train_losses,
        val_losses,
        map50s, mar100s, f1s,
        maps, map75s, mar1s, mar10s,
        m_s, m_m, m_l,
        r_s, r_m, r_l
    ) = metrics_out

    # Plot key curves
    plot_curve(train_losses, "Train Loss",  "Train Loss",  os.path.join(result_dir, "train_loss.png"))
    plot_curve(val_losses,   "Val Loss",    "Val Loss",    os.path.join(result_dir, "val_loss.png"))
    plot_curve(map50s,       "mAP@.50",     "mAP@.50",     os.path.join(result_dir, "map50.png"))
    plot_curve(mar100s,      "mAR@100",     "mAR@100",     os.path.join(result_dir, "mar100.png"))

    # Save everything in metrics.csv
    metrics_dict = {
        "epoch":          list(range(1, len(train_losses) + 1)),
        "train_loss":     train_losses,
        "val_loss":       val_losses,
        "mAP@.50":        map50s,
        "mAR@100":        mar100s,
        "F1":             f1s,
        "mAP@[.50:.95]":  maps,
        "mAP@.75":        map75s,
        "mAR@1":          mar1s,
        "mAR@10":         mar10s,
        "mAP_small":      m_s,
        "mAP_medium":     m_m,
        "mAP_large":      m_l,
        "mAR_small":      r_s,
        "mAR_medium":     r_m,
        "mAR_large":      r_l,
    }
    save_metrics_csv(os.path.join(log_dir, "metrics.csv"), metrics_dict)

    # Save model weights
    ckpt_path = os.path.join(ckpt_dir, f"{model_type}_model.pth")
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Model checkpoint saved at: {ckpt_path}")

    # Optional: run test‐set visualization if present
    if os.path.isdir(test_imgs) and os.path.isdir(test_lbls):
        test_ds = TestDataset(
            test_imgs, test_lbls,
            cfg["classes_file"],
            transforms=transform,
            normalize_boxes=is_detr
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda x: x[0]
        )
        out_vis = os.path.join(result_dir, "test_results.png")
        evaluate_and_visualize(
            model, test_loader, classes, device, out_vis,
            threshold=cfg.get("eval_threshold", 0.5),
            model_type=model_type
        )
    else:
        logger.info("Test set not found; skipping inference‐viz.")


if __name__ == "__main__":
    main()
