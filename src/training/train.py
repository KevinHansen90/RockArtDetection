#!/usr/bin/env python3
import os

# Force MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import sys
import tempfile

# Ensure project root on sys.path
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

# -- Optional GCS helpers -----------------------------------------
try:
	from google.cloud import storage
except ImportError:
	storage = None


def download_blob(gcs_uri: str, dst_path: str):
	bucket_name, blob_name = gcs_uri[5:].split("/", 1)
	client = storage.Client()
	bucket = client.bucket(bucket_name)
	blob = bucket.blob(blob_name)
	blob.download_to_filename(dst_path)


def download_dir(gcs_uri: str, local_dir: str):
	bucket_name, prefix = gcs_uri[5:].split("/", 1)
	client = storage.Client()
	blobs = client.list_blobs(bucket_name, prefix=prefix)
	for blob in blobs:
		rel_path = blob.name[len(prefix):].lstrip("/")
		dst_path = os.path.join(local_dir, rel_path)
		os.makedirs(os.path.dirname(dst_path), exist_ok=True)
		blob.download_to_filename(dst_path)


def upload_dir(local_dir: str, gcs_uri: str):
	bucket_name, prefix = gcs_uri[5:].split("/", 1)
	client = storage.Client()
	bucket = client.bucket(bucket_name)
	for root, _, files in os.walk(local_dir):
		for fname in files:
			local_path = os.path.join(root, fname)
			rel_path = os.path.relpath(local_path, local_dir)
			blob = bucket.blob(f"{prefix}/{rel_path}")
			blob.upload_from_filename(local_path)


# ------------------------------------------------------------------
# Logger setup
logger = logging.getLogger("train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
	parser = argparse.ArgumentParser(description="Train an object detection model")
	parser.add_argument("--config", required=True, help="Path to YAML config (local or gs://)")
	parser.add_argument("--experiment", required=True, help="Experiment name")
	parser.add_argument("--output-dir", default=None,
						help="Base output directory (local or gs://) for artifacts")
	opts = parser.parse_args()

	# --- Stage 0: Handle config download if on GCS ----------------
	cfg_path = opts.config
	if cfg_path.startswith("gs://"):
		assert storage, "google-cloud-storage is required to load remote configs"
		tmp_cfg = os.path.join(tempfile.gettempdir(), opts.experiment, "used_config.yaml")
		os.makedirs(os.path.dirname(tmp_cfg), exist_ok=True)
		download_blob(cfg_path, tmp_cfg)
		cfg_path = tmp_cfg

	# Load config + set seeds
	cfg = load_config(cfg_path)
	seed = cfg.get("seed", 42)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

	# --- Stage 1: Determine output base directory ------------------
	use_cloud = False
	out_base = opts.output_dir
	# Override with Vertex AI env var if present
	aip_model = os.getenv("AIP_MODEL_DIR")
	if aip_model:
		use_cloud = True
		out_base = aip_model.rsplit('/', 1)[0]

	if out_base:
		if out_base.startswith("gs://"):
			use_cloud = True
			local_base = os.path.join(tempfile.gettempdir(), opts.experiment)
		else:
			local_base = out_base
	else:
		local_base = os.path.join("experiments", opts.experiment)

	# Prepare experiment dirs locally
	exp_dir = local_base
	ckpt_dir = os.path.join(exp_dir, "checkpoints")
	log_dir = os.path.join(exp_dir, "logs")
	result_dir = os.path.join(exp_dir, "results")
	for d in (exp_dir, ckpt_dir, log_dir, result_dir):
		os.makedirs(d, exist_ok=True)

	# Copy used config
	dest_cfg = os.path.join(exp_dir, "used_config.yaml")
	if not cfg_path.endswith(dest_cfg):
		shutil.copy2(cfg_path, dest_cfg)
	logger.info(f"Experiment directory: {exp_dir}")

	# --- Stage 2: Handle remote classes_file & data_root ----------
	classes_file = cfg["classes_file"]
	if classes_file.startswith("gs://"):
		tmp_cls = os.path.join(tempfile.gettempdir(), opts.experiment, "grouped_classes.txt")
		os.makedirs(os.path.dirname(tmp_cls), exist_ok=True)
		download_blob(classes_file, tmp_cls)
		classes_file = tmp_cls
	data_root = cfg["data_root"]
	if data_root.startswith("gs://"):
		tmp_data = os.path.join(tempfile.gettempdir(), opts.experiment, "data_root")
		download_dir(data_root, tmp_data)
		data_root = tmp_data

	# After downloading, override cfg for downstream code:
	cfg["classes_file"] = classes_file
	cfg["data_root"] = data_root

	# Device
	device = get_device()
	logger.info(f"Using device: {device}")

	# Load classes and data paths
	classes = load_classes(classes_file)
	train_imgs = os.path.join(data_root, "train/images")
	train_lbls = os.path.join(data_root, "train/labels")
	val_imgs = os.path.join(data_root, "val/images")
	val_lbls = os.path.join(data_root, "val/labels")
	test_imgs = os.path.join(data_root, "test/images")
	test_lbls = os.path.join(data_root, "test/labels")

	# Dataset & Dataloader
	is_detr = (cfg["model_type"].lower() == "deformable_detr")
	requires_shift_for_viz = not is_detr
	coll = collate_fn_detr if is_detr else collate_fn
	transform = get_simple_transform()

	train_ds = CustomYOLODataset(train_imgs, train_lbls, classes_file, transform, is_detr)
	val_ds = CustomYOLODataset(val_imgs, val_lbls, classes_file, transform, is_detr)

	g = torch.Generator().manual_seed(seed)
	dl_kwargs = {
		"batch_size": cfg.get("batch_size", 2),
		"shuffle": True,
		"num_workers": cfg.get("num_workers", 0),
		"pin_memory": (device.type == "cuda"),
		"collate_fn": coll,
		"generator": g,
	}
	if cfg.get("num_workers", 0) > 0:
		dl_kwargs.update({"prefetch_factor": 2, "persistent_workers": True})

	train_loader = DataLoader(train_ds, **dl_kwargs)
	val_loader = DataLoader(val_ds, **{**dl_kwargs, "shuffle": False})

	# Model, optimizer, scheduler
	model_type = cfg["model_type"].lower()
	num_classes = len(classes) + (0 if is_detr else 1)
	model = get_detection_model(model_type, num_classes, config=cfg).to(device)
	# optional: compile + fuse for faster kernels on GPU (no effect on CPU)
	if device.type == "cuda":
		model = torch.compile(model)
	optim = get_optimizer(model, cfg)
	sched = get_scheduler(optim, cfg)

	cfg["log_dir"] = log_dir

	# Train & collect metrics
	metrics_out = train_model(model, train_loader, val_loader, device, optim, sched, cfg)
	(train_losses, val_losses, map50s, mar100s, f1s,
	 maps, map75s, mar1s, mar10s, m_s, m_m, m_l,
	 r_s, r_m, r_l) = metrics_out

	# Plot key curves
	plot_curve(train_losses, "Train Loss", "Train Loss", os.path.join(result_dir, "train_loss.png"))
	plot_curve(val_losses, "Val Loss", "Val Loss", os.path.join(result_dir, "val_loss.png"))
	plot_curve(map50s, "mAP@.50", "mAP@.50", os.path.join(result_dir, "map50.png"))
	plot_curve(mar100s, "mAR@100", "mAR@100", os.path.join(result_dir, "mar100.png"))

	# Save metrics CSV
	metrics_dict = {
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
	}
	save_metrics_csv(os.path.join(log_dir, "metrics.csv"), metrics_dict)

	# Save model checkpoint
	ckpt_path = os.path.join(ckpt_dir, f"{model_type}_model.pth")
	torch.save(model.state_dict(), ckpt_path)
	logger.info(f"Model checkpoint saved at: {ckpt_path}")

	# Optional: run test‐set visualization
	if os.path.isdir(test_imgs) and os.path.isdir(test_lbls):
		test_ds = TestDataset(test_imgs, test_lbls,
							  classes_file, transforms=transform,
							  normalize_boxes=is_detr,
							  shift_labels=requires_shift_for_viz)
		test_loader = DataLoader(test_ds,
								 batch_size=1, shuffle=False,
								 num_workers=0,
								 collate_fn=lambda x: x[0])
		out_vis = os.path.join(result_dir, "test_results.png")
		evaluate_and_visualize(
			model, test_loader, classes, device, out_vis,
			threshold=cfg.get("eval_threshold", 0.5),
			model_type=model_type
		)
	else:
		logger.info("Test set not found; skipping inference‐viz.")

	# If using cloud, upload all artifacts at end
	if use_cloud and out_base and out_base.startswith("gs://"):
		upload_dir(ckpt_dir, f"{out_base}/checkpoints")
		upload_dir(log_dir, f"{out_base}/logs")
		upload_dir(result_dir, f"{out_base}/results")
		bucket_name, prefix = out_base[5:].split("/", 1)
		client = storage.Client()
		blob = client.bucket(bucket_name).blob(f"{prefix}/used_config.yaml")
		blob.upload_from_filename(dest_cfg)
		logger.info(f"Uploaded experiment results to {out_base}")


if __name__ == "__main__":
	main()