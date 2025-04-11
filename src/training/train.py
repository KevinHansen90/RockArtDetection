#!/usr/bin/env python
# src/training/train.py

import os

# IMPORTANT: Enable MPS fallback (must happen before torch import)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import shutil
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.utils import load_config, get_device, get_simple_transform, plot_curve, save_metrics_csv, freeze_batchnorm
from src.training.engine import train_model
from src.training.evaluate import evaluate_and_visualize
from src.datasets.yolo_dataset import CustomYOLODataset, TestDataset, collate_fn, collate_fn_detr, load_classes
from src.models.detection_models import get_detection_model


def get_optimizer(model, config):
    freeze_backbone = config.get("freeze_backbone", False)

    if hasattr(model, "hf_model"):
        backbone_params = list(model.hf_model.model.backbone.parameters())
    elif hasattr(model, "model"):
        backbone_params = list(model.model.backbone.parameters())
    else:
        backbone_params = []

    if freeze_backbone:
        # Freeze all backbone parameters initially and log their names.
        for name, param in model.backbone.named_parameters():
            param.requires_grad = False
        # Only include parameters with requires_grad == True (i.e., non-backbone parameters) in the optimizer.
        optimizer_params = [{"params": [p for p in model.parameters() if p.requires_grad],
                             "lr": config.get("head_lr", 5e-4)}]
    else:
        # Include both backbone and head parameters.
        all_params = list(model.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        other_params = [p for p in all_params if id(p) not in backbone_ids]
        optimizer_params = [
            {"params": backbone_params, "lr": config.get("backbone_lr", 5e-5)},
            {"params": other_params, "lr": config.get("head_lr", 5e-4)}
        ]

    optimizer_choice = config.get("optimizer", "AdamW").lower()
    if optimizer_choice == "sgd":
        optimizer = torch.optim.SGD(optimizer_params, momentum=config.get("momentum", 0.9), weight_decay=config.get("weight_decay", 0.0005))
    elif optimizer_choice == "adam":
        optimizer = torch.optim.Adam(optimizer_params, weight_decay=config.get("weight_decay", 0.0005))
    elif optimizer_choice == "adamw":
        optimizer = torch.optim.AdamW(optimizer_params, weight_decay=config.get("weight_decay", 0.0005), eps=config.get("eps", 0.0000001))
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_choice}")
    return optimizer


def get_scheduler(optimizer, config):
    """Create a scheduler if specified in config."""
    import torch
    sched_type = config.get("scheduler", None)
    if not sched_type or sched_type.lower() == "none":
        return None
    elif sched_type == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 7),
            gamma=config.get("gamma", 0.1)
        )
    elif sched_type == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=config.get("factor", 0.1),
            patience=config.get("plateau_patience", 5)
        )
    elif sched_type == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get("T_max", 10),  # Consider setting to epochs
            eta_min=config.get("eta_min", 0)
        )
    elif sched_type == "CosineAnnealingWarmRestarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.get("T_0", 10),  # Number of epochs for the first restart
            T_mult=config.get("T_mult", 1),  # Multiplication factor for next restart
            eta_min=config.get("eta_min", 0)
        )
    elif sched_type == "MultiStepLR":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.get("milestones", [30, 60]),  # List of epoch indices
            gamma=config.get("gamma", 0.1)
        )
    elif sched_type == "OneCycleLR":
        # If max_lr not in config, defaults to optimizer's initial lr
        max_lr = config.get("max_lr", [pg['lr'] for pg in optimizer.param_groups])
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=config.get("total_steps", 100),  # Must specify in config
            pct_start=config.get("pct_start", 0.3),
            anneal_strategy=config.get("anneal_strategy", "cos"),
            cycle_momentum=config.get("cycle_momentum", True),
            base_momentum=config.get("base_momentum", 0.85),
            max_momentum=config.get("max_momentum", 0.95),
            div_factor=config.get("div_factor", 25.0),
            final_div_factor=config.get("final_div_factor", 10000.0)
        )
    else:
        raise ValueError(f"Unsupported scheduler: {sched_type}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a TorchVision Detection Model.")
    parser.add_argument("--config", type=str, default="configs/detection_config.yaml",
                        help="Path to your YAML config file.")
    parser.add_argument("--experiment", type=str, default="default_experiment",
                        help="Name for this experiment (subfolder in 'experiments/').")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config
    exp_name = args.experiment

    # 1. Load config
    config = load_config(config_path)

    # 2. Set random seed for reproducibility
    seed = config.get("seed", 42)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Define worker seeding function
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    # Create generator with fixed seed
    g = torch.Generator()
    g.manual_seed(seed)

    # 3. Prepare experiment folder
    experiment_dir = os.path.join("experiments", exp_name)
    os.makedirs(experiment_dir, exist_ok=True)
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    logs_dir = os.path.join(experiment_dir, "logs")
    results_dir = os.path.join(experiment_dir, "results")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    # Optionally copy the config file into the experiment folder for reference
    shutil.copy2(config_path, os.path.join(experiment_dir, "used_config.yaml"))

    # 4. Prepare device
    device = get_device()
    tqdm.write(f"Using device: {device}")

    # 5. Setup dataset paths
    data_root = config["data_root"]
    train_images = os.path.join(data_root, "train/images")
    train_labels = os.path.join(data_root, "train/labels")
    val_images = os.path.join(data_root, "val/images")
    val_labels = os.path.join(data_root, "val/labels")
    test_images = os.path.join(data_root, "test/images")
    test_labels = os.path.join(data_root, "test/labels")
    classes_file = config["classes_file"]
    classes = load_classes(classes_file)

    # Determine if we need to normalize boxes (for deformable_detr)
    model_type = config.get("model_type", "fasterrcnn").lower()
    normalize_boxes = (model_type == "deformable_detr")

    if model_type == "deformable_detr":
        num_classes = len(classes)  # no extra background class
    else:
        num_classes = len(classes) + 1  # +1 for background

    # 6. Create datasets & loaders (pass normalize_boxes flag accordingly)
    transform_fn = get_simple_transform()
    train_dataset = CustomYOLODataset(train_images, train_labels, classes_file, transforms=transform_fn, normalize_boxes=normalize_boxes)
    val_dataset = CustomYOLODataset(val_images, val_labels, classes_file, transforms=transform_fn, normalize_boxes=normalize_boxes)

    if model_type == "deformable_detr":
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                                  shuffle=True, num_workers=config["num_workers"],
                                  collate_fn=collate_fn_detr)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"],
                                shuffle=False, num_workers=config["num_workers"],
                                collate_fn=collate_fn_detr)
    else:
        # Use the default collate_fn for other models.
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                                  shuffle=True, num_workers=config["num_workers"],
                                  collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"],
                                shuffle=False, num_workers=config["num_workers"],
                                collate_fn=collate_fn)

    # 7. Build model, optimizer, scheduler
    model = get_detection_model(model_type, num_classes, config=config).to(device)

    if model_type == "retinanet":
        # If we are freezing the entire backbone, we don't need to freeze only BN layers.
        if not config.get("freeze_backbone", False):
            freeze_batchnorm(model)
        model.score_thresh = config.get("eval_threshold", 0.1)

    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # 8. Train
    train_losses, val_losses, map_list, mar_list = train_model(
        model, train_loader, val_loader, device, optimizer, scheduler, config
    )

    # 9. Plot metrics
    train_loss_path = os.path.join(results_dir, "train_loss.png")
    val_loss_path = os.path.join(results_dir, "val_loss.png")
    map_evol_path = os.path.join(results_dir, "map_evolution.png")
    mar_evol_path = os.path.join(results_dir, "mar_evolution.png")
    plot_curve(train_losses, "Train Loss", "Train Loss Curve", train_loss_path)
    plot_curve(val_losses, "Val Loss", "Val Loss Curve", val_loss_path)
    plot_curve(map_list, "mAP@0.5", "mAP Evolution", map_evol_path)
    plot_curve(mar_list, "mAR@100", "mAR Evolution", mar_evol_path)

    # 10. Save csv
    csv_path = os.path.join(logs_dir, "metrics.csv")
    save_metrics_csv(csv_path, train_losses, val_losses, map_list, mar_list)
    tqdm.write(f"Saved CSV metrics to: {csv_path}")

    # 11. Save checkpoint
    model_ckpt_path = os.path.join(checkpoints_dir, f"{model_type}_model.pth")
    torch.save(model.state_dict(), model_ckpt_path)
    tqdm.write(f"Model saved to: {model_ckpt_path}")

    # 12. Evaluate on test set (if exists)
    if os.path.exists(test_images) and os.path.exists(test_labels):
        test_dataset = TestDataset(test_images, test_labels, classes_file, transforms=transform_fn, normalize_boxes=normalize_boxes)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, worker_init_fn=seed_worker,
                                 collate_fn=lambda x: x[0])
        test_output_path = os.path.join(results_dir, "test_results.png")
        threshold = config.get("eval_threshold", 0.5)
        evaluate_and_visualize(model, test_loader, classes, device, test_output_path, threshold=threshold, model_type=model_type)
    else:
        tqdm.write("No test dataset found. Skipping test evaluation.")


if __name__ == "__main__":
    main()

