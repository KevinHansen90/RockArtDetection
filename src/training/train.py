#!/usr/bin/env python3
from __future__ import annotations

import argparse, logging, os, random, shutil, sys, tempfile, threading
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------- project imports ----------------------------------------- #
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT)) if str(_ROOT) not in sys.path else None

from src.training.utils import (
    load_config, get_device, get_simple_transform, get_train_transform,
    plot_curve, save_metrics_csv, setup_logging, DEFAULT_NUM_WORKERS,
)
from src.training.engine import train_model
from src.models.detection_models import get_detection_model, get_optimizer, get_scheduler
from src.datasets.yolo_dataset import YOLODataset, collate_fn, collate_fn_detr, load_classes
from src.training.evaluate import evaluate_and_visualize

# ---------------- cloud helpers ------------------------------------------- #
try:
    from google.cloud import storage  # type: ignore
except ImportError:
    storage = None  # pragma: no cover

# ---------------- logging -------------------------------------------------- #
setup_logging()
log = logging.getLogger("train")


# -------------------------------------------------------------------------- #
# GCS helpers (unchanged bodies)                                             #
# -------------------------------------------------------------------------- #
def _download_blob(gcs_uri: str, dst_path: str) -> None:
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(dst_path)


def _download_dir(gcs_uri: str, local_dir: str) -> None:
    bucket, pref = gcs_uri[5:].split("/", 1)
    client = storage.Client()
    for blob in client.list_blobs(bucket, prefix=pref):
        rel = Path(blob.name[len(pref):].lstrip("/"))
        dst = Path(local_dir, rel); dst.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(dst)


def _upload_dir(local_dir: str, gcs_uri: str) -> None:
    bucket, pref = gcs_uri[5:].split("/", 1)
    client = storage.Client()
    for root, _, files in os.walk(local_dir):
        for f in files:
            lp = Path(root, f); rel = lp.relative_to(local_dir)
            client.bucket(bucket).blob(f"{pref}/{rel}").upload_from_filename(lp)


# -------------------------------------------------------------------------- #
# Arg-parser                                                                 #
# -------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("train.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", required=True, help="Path to YAML config (local or gs://)")
    p.add_argument("--experiment", required=True, help="Experiment name")
    p.add_argument("--output-dir", default=None,
                   help="Base output directory (local or gs://) for artifacts")
    p.add_argument("--compile", action="store_true",
                   help="Enable torch.compile (CUDA only).")
    p.add_argument("--deterministic", action="store_true",
                   help="Force deterministic algorithms & seeds.")
    p.add_argument("--async-upload", action="store_true",
                   help="Upload checkpoints/logs to GCS in background thread.")
    return p


# worker-seed helper for full determinism
def _seed_worker(worker_id: int, base_seed: int):
    ws = base_seed + worker_id
    np.random.seed(ws); random.seed(ws)


# -------------------------------------------------------------------------- #
# Main                                                                       #
# -------------------------------------------------------------------------- #
def main(argv: List[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    cfg_path = args.config

    # --- fetch remote YAML ------------------------------------------------ #
    if cfg_path.startswith("gs://"):
        tmp = Path(tempfile.gettempdir(), args.experiment, "used_config.yaml")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        _download_blob(cfg_path, tmp)
        cfg_path = str(tmp)

    cfg: Dict[str, Any] = load_config(cfg_path)
    cfg["compile"] = args.compile  # so model factory can read it

    # ---------------------------------------------------------------------- #
    # Reproducibility seeding                                                #
    # ---------------------------------------------------------------------- #
    seed = cfg.get("seed", 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        log.info("Deterministic mode enabled.")

    # ---------------------------------------------------------------------- #
    # Experiment directories                                                 #
    # ---------------------------------------------------------------------- #
    use_cloud, out_base = False, args.output_dir
    aip_model = os.getenv("AIP_MODEL_DIR")
    if aip_model:
        out_base, use_cloud = str(Path(aip_model).parent), True

    if out_base and out_base.startswith("gs://"):
        use_cloud, local_base = True, Path(tempfile.gettempdir(), args.experiment)
    else:
        local_base = Path(out_base) if out_base else Path("experiments", args.experiment)

    ckpt_dir, log_dir, result_dir = [local_base / d for d in ("checkpoints", "logs", "results")]
    for d in (ckpt_dir, log_dir, result_dir): d.mkdir(parents=True, exist_ok=True)

    dest_cfg = local_base / "used_config.yaml"
    if not str(cfg_path).endswith(str(dest_cfg)): shutil.copy2(cfg_path, dest_cfg)
    log.info("Experiment directory: %s", local_base)

    # ---------------------------------------------------------------------- #
    # Remote classes / dataset download                                      #
    # ---------------------------------------------------------------------- #
    classes_file = cfg["classes_file"]
    if classes_file.startswith("gs://"):
        tmp_cls = Path(tempfile.gettempdir(), args.experiment, "classes.txt")
        tmp_cls.parent.mkdir(parents=True, exist_ok=True)
        _download_blob(classes_file, tmp_cls)
        classes_file = str(tmp_cls)

    data_root = cfg["data_root"]
    if data_root.startswith("gs://"):
        dr = Path(tempfile.gettempdir(), args.experiment, "data_root")
        _download_dir(data_root, dr)
        data_root = str(dr)

    cfg.update({"classes_file": classes_file, "data_root": data_root})

    # ---------------------------------------------------------------------- #
    device = get_device(); log.info("Using device: %s", device)
    classes = load_classes(classes_file)
    paths = lambda split: (Path(data_root, f"{split}/images"),
                           Path(data_root, f"{split}/labels"))
    train_imgs, train_lbls = paths("train")
    val_imgs, val_lbls     = paths("val")
    test_imgs, test_lbls   = paths("test")

    is_detr  = cfg["model_type"].lower() == "deformable_detr"
    collate  = collate_fn_detr if is_detr else collate_fn
    mps_safe = device.type == "mps"

    train_tf = get_train_transform(is_detr, mps_safe, seed)
    test_tf  = get_simple_transform()

    train_ds = YOLODataset(train_imgs, train_lbls, classes_file,
                           mode="train", transforms=train_tf,
                           normalize_boxes=is_detr, shift_labels=not is_detr)
    val_ds   = YOLODataset(val_imgs,   val_lbls,   classes_file,
                           mode="val",  transforms=test_tf,
                           normalize_boxes=is_detr, shift_labels=not is_detr)

    workers = cfg.get("num_workers", DEFAULT_NUM_WORKERS)
    dl_kwargs = dict(batch_size=cfg.get("batch_size", 2),
                     num_workers=workers,
                     pin_memory=(device.type == "cuda"),
                     collate_fn=collate,
                     shuffle=True)
    if args.deterministic:
        dl_kwargs.update(worker_init_fn=lambda wid: _seed_worker(wid, seed),
                         generator=torch.Generator().manual_seed(seed))
    if workers:
        dl_kwargs.update(prefetch_factor=2, persistent_workers=True)

    train_loader = DataLoader(train_ds, **dl_kwargs)
    val_loader   = DataLoader(val_ds,   **{**dl_kwargs, "shuffle": False})

    # ---------------------------------------------------------------------- #
    model_type = cfg["model_type"].lower()
    num_classes = len(classes) + (0 if is_detr else 1)
    model = get_detection_model(model_type, num_classes, cfg)

    run_device = torch.device("cpu") if (device.type == "mps" and model_type == "fasterrcnn") else device
    if run_device.type == "cuda" and cfg["compile"]:
        model = torch.compile(model)
    model = model.to(run_device)

    optim = get_optimizer(model, cfg)
    sched = get_scheduler(optim, cfg)
    cfg["log_dir"] = str(log_dir)

    # ---------------------------------------------------------------------- #
    metrics = train_model(model, train_loader, val_loader, run_device, optim, sched, cfg)
    (train_losses, val_losses,
     map50s, mar100s, f1s,
     maps, map75s, mar1s, mar10s,
     m_s, m_m, m_l,
     r_s, r_m, r_l) = metrics

    # ---------------- plotting + CSV -------------------------------------- #
    plot_curve(train_losses, "Train Loss", "Train Loss", result_dir / "train_loss.png")
    plot_curve(val_losses,   "Val Loss",   "Val Loss",   result_dir / "val_loss.png")
    plot_curve(map50s,       "mAP@.50",    "mAP@.50",    result_dir / "map50.png")
    plot_curve(mar100s,      "mAR@100",    "mAR@100",    result_dir / "mar100.png")

    save_metrics_csv(log_dir / "metrics.csv", {
        "epoch": list(range(1, len(train_losses)+1)),
        "train_loss": train_losses, "val_loss": val_losses,
        "mAP@.50": map50s, "mAR@100": mar100s, "F1": f1s,
        "mAP@[.50:.95]": maps, "mAP@.75": map75s,
        "mAR@1": mar1s, "mAR@10": mar10s,
        "mAP_small": m_s, "mAP_medium": m_m, "mAP_large": m_l,
        "mAR_small": r_s, "mAR_medium": r_m, "mAR_large": r_l,
    })

    # ---------------- checkpoint & vis ------------------------------------ #
    ckpt = ckpt_dir / f"{model_type}_model.pth"
    torch.save(model.state_dict(), ckpt); log.info("Saved model → %s", ckpt)

    if test_imgs.is_dir() and test_lbls.is_dir():
        vis_ds = YOLODataset(test_imgs, test_lbls, classes_file,
                             mode="test", transforms=test_tf,
                             normalize_boxes=is_detr, shift_labels=not is_detr)
        vis_loader = DataLoader(vis_ds, batch_size=1, shuffle=False, num_workers=0,
                                collate_fn=lambda x: x[0])
        evaluate_and_visualize(model, vis_loader, classes, run_device,
                               str(result_dir / "test_results.png"),
                               threshold=cfg.get("eval_threshold", 0.5),
                               model_type=model_type)

    # ---------------- optional async cloud upload ------------------------- #
    def _async_upload():
        _upload_dir(ckpt_dir,  f"{out_base}/checkpoints")
        _upload_dir(log_dir,   f"{out_base}/logs")
        _upload_dir(result_dir,f"{out_base}/results")
        storage.Client().bucket(out_base[5:].split('/',1)[0]).blob(
            f"{out_base.split('/',1)[1]}/used_config.yaml").upload_from_filename(dest_cfg)
        log.info("Uploaded artifacts to %s", out_base)

    if use_cloud and args.async_upload and str(out_base).startswith("gs://"):
        threading.Thread(target=_async_upload, daemon=True).start()
    elif use_cloud and str(out_base).startswith("gs://"):
        _async_upload()


if __name__ == "__main__":
    main()

