#!/usr/bin/env python3
from __future__ import annotations

# ── env flags ─────────────────────────────────────────────────────────
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# ── std-lib / third-party ─────────────────────────────────────────────
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

# ── project root on path ──────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── local helpers ─────────────────────────────────────────────────────
from src.training.utils import (
    DEFAULT_NUM_WORKERS,
    get_cfg_dict,
    get_device,
    get_simple_transform,
    plot_curve,
    save_metrics_csv,
    setup_logging,
)
from src.transforms import build_train as _build_train_tf
from src.training.engine import train_model
from src.models.detection_models import get_detection_model, get_optimizer, get_scheduler
from src.datasets.yolo_dataset import (
    YOLODataset,
    collate_fn,
    collate_fn_detr,
    load_classes,
)
from src.training.evaluate import evaluate_and_visualize

setup_logging()
log = logging.getLogger("train")

# ══════════════════════════════════════════════════════════════════════
def run_training(cfg: Dict[str, Any]) -> None:
    # 0 ─── W&B optional ───────────────────────────────────────────────
    wb_cfg = cfg.get("wandb", {})
    wandb_enabled = wb_cfg.get("enabled", False) and wandb is not None
    wandb_run = None
    if wandb_enabled:
        wandb_run = wandb.init(
            project=wb_cfg.get("project", "rockart-detection"),
            entity=wb_cfg.get("entity"),
            name=wb_cfg.get("name"),
            mode=wb_cfg.get("mode", "online"),
            config=cfg,
        )

    # 1 ─── Reproducibility ────────────────────────────────────────────
    seed = cfg.get("seed", 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2 ─── Experiment dirs ────────────────────────────────────────────
    run_name = cfg.get("experiment", "run")
    vertex_out = os.getenv("AIP_STORAGE_URI") or os.getenv("AIP_MODEL_DIR")
    if vertex_out and vertex_out.startswith("gs://"):
        vertex_out = "/gcs/" + vertex_out[len("gs://") :]
    exp_root = Path(vertex_out) if vertex_out else _ROOT / "experiments" / run_name
    os.environ["AIP_MODEL_DIR"] = os.environ["AIP_STORAGE_URI"] = str(exp_root)

    ckpt_dir, log_dir, result_dir = [exp_root / s for s in ("checkpoints", "logs", "results")]
    for d in (ckpt_dir, log_dir, result_dir):
        d.mkdir(parents=True, exist_ok=True)
    log.info("Experiment dir: %s", exp_root)

    # snapshot merged config
    try:
        OmegaConf.save(OmegaConf.create(cfg), exp_root / "used_config.yaml")
    except Exception as e:
        log.warning("Could not save config snapshot: %s", e)

    # 3 ─── Dataset paths ──────────────────────────────────────────────
    data_root, classes_file = cfg["data_root"], cfg["classes_file"]
    classes = load_classes(classes_file)

    def split(part: str):
        if data_root.startswith("gs://"):
            return f"{data_root}/{part}/images", f"{data_root}/{part}/labels"
        return Path(data_root) / part / "images", Path(data_root) / part / "labels"

    train_imgs, train_lbls = split("train")
    val_imgs, val_lbls = split("val")
    test_imgs, test_lbls = split("test")

    # 4 ─── Device & transforms ────────────────────────────────────────
    device = get_device()
    if cfg["model_type"].lower() == "fasterrcnn" and device.type == "mps":
        log.warning("Faster R-CNN not fully supported on MPS – falling back to CPU.")
        device = torch.device("cpu")
    log.info("Device: %s", device)

    is_detr = cfg["model_type"].lower() == "deformable_detr"
    train_tf = _build_train_tf(is_detr, device, seed)
    test_tf = get_simple_transform()

    # 5 ─── Datasets / loaders ─────────────────────────────────────────
    train_ds = YOLODataset(
        train_imgs, train_lbls, classes_file, mode="train",
        transforms=train_tf, normalize_boxes=is_detr, shift_labels=not is_detr,
    )
    val_ds = YOLODataset(
        val_imgs, val_lbls, classes_file, mode="val",
        transforms=test_tf, normalize_boxes=is_detr, shift_labels=not is_detr,
    )
    workers = cfg.get("num_workers", DEFAULT_NUM_WORKERS)
    collate = collate_fn_detr if is_detr else collate_fn

    tf_dev = getattr(train_tf, "_tv2_device", torch.device("cpu"))
    _pin_memory = device.type == "cuda" and tf_dev.type == "cpu"
    dl_kw = dict(
        batch_size=cfg.get("batch_size", 2),
        shuffle=True,
        num_workers=workers,
        pin_memory=_pin_memory,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
    )
    if workers:
        dl_kw.update(prefetch_factor=2, persistent_workers=True)
    train_loader = DataLoader(train_ds, **dl_kw)
    val_loader = DataLoader(val_ds, **{**dl_kw, "shuffle": False})

    # 6 ─── Model / optim / sched ──────────────────────────────────────
    num_classes = len(classes) + (0 if is_detr else 1)
    model = get_detection_model(cfg["model_type"], num_classes, cfg).to(device)
    optim = get_optimizer(model, cfg)
    sched = get_scheduler(optim, cfg)
    cfg["log_dir"] = str(log_dir)

    # 7 ─── Training ───────────────────────────────────────────────────
    (
        train_losses, val_losses, map50s, mar100s, f1s,
        maps, map75s, mar1s, mar10s,
        m_s, m_m, m_l, r_s, r_m, r_l,
        lr0s, lr1s, epoch_times,
    ) = train_model(
        model, train_loader, val_loader, device,
        optim, sched, cfg, wandb_run=wandb_run,
    )

    # 8 ─── Plots & CSV ────────────────────────────────────────────────
    plot_curve(train_losses, "Train Loss", "Train Loss", result_dir / "train_loss.png")
    plot_curve(val_losses, "Val Loss", "Val Loss", result_dir / "val_loss.png")
    plot_curve(map50s, "mAP@.50", "mAP@.50", result_dir / "map50.png")
    plot_curve(mar100s, "mAR@100", "mAR@100", result_dir / "mar100.png")

    save_metrics_csv(
        result_dir / "metrics.csv",
        {
            "epoch": list(range(1, len(train_losses) + 1)),
            "train_loss": train_losses,
            "val_loss": val_losses,
            "mAP@.50": map50s,
            "mAR@100": mar100s,
            "F1": f1s,
            "mAP@[.50:.95]": maps,
            "mAP@.75": map75s,
            "mAR@1": mar1s,
            "mAR@10": mar10s,
            "mAP_small": m_s,
            "mAP_medium": m_m,
            "mAP_large": m_l,
            "mAR_small": r_s,
            "mAR_medium": r_m,
            "mAR_large": r_l,
            "lr_group0": lr0s,
            "lr_group1": lr1s,
            "epoch_time": epoch_times,
        },
    )

    # 9 ─── Checkpoint & test collage ──────────────────────────────────
    torch.save(model.state_dict(), ckpt_dir / f"{cfg['model_type']}_model.pth")

    stream = str(test_imgs).startswith("gs://")

    test_ds = YOLODataset(
        test_imgs, test_lbls, classes_file, mode="test",
        transforms=test_tf, normalize_boxes=is_detr, shift_labels=not is_detr,
        stream_gcs=stream,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0,
                             collate_fn=lambda x: x[0])
    evaluate_and_visualize(
        model, test_loader, classes, device,
        result_dir / "results.png",
        threshold=cfg.get("eval_threshold", 0.5),
        model_type=cfg["model_type"],
    )

    # 10 ─── Close W&B & clean cache ───────────────────────────────────
    if wandb_run is not None:
        wandb_run.finish()
        wandb_dir = Path(os.environ.get("WANDB_DIR", "./wandb")).resolve()
        if wandb_dir.exists() and not str(wandb_dir).startswith("/gcs/"):
            shutil.rmtree(wandb_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════
@hydra.main(config_path=str(_ROOT / "configs"), config_name="defaults", version_base="1.3")
def main(cfg: DictConfig) -> None:
    flat = get_cfg_dict(cfg)
    flat["experiment"] = cfg.get("experiment", "run")
    run_training(flat)


if __name__ == "__main__":  # pragma: no cover
    main()

