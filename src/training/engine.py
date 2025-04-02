# src/training/engine.py

import torch
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision
from src.training.utils import compute_total_loss
from src.models.detection_models import DeformableDETRWrapper


def evaluate_on_dataset(model, data_loader, device, verbose=False):
    """
    Compute average validation loss, mAP@0.5, and mAR@100 using a two-pass approach.
    """
    # Pass 1: measure validation loss
    model.train()  # use train() so that dropout is active if needed, but disable gradients
    total_loss = 0.0
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            loss_dict = model(images, targets)
            if verbose and batch_idx == 0:
                loss_details = ", ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
                tqdm.write(f"[Validation Batch {batch_idx}] Loss breakdown: {loss_details}")
        total_loss += compute_total_loss(loss_dict).item()
    avg_loss = total_loss / len(data_loader)

    # Pass 2: measure predictions for mAP/mAR
    model.eval()
    map_metric = MeanAveragePrecision()
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            preds = model(images)
        # If using a special wrapper (e.g. Deformable DETR), adjust targets.
        if isinstance(model, DeformableDETRWrapper):
            processed_targets = []
            for t in targets:
                h, w = t["orig_size"][0].item(), t["orig_size"][1].item()
                cx = t["boxes"][:, 0] * w
                cy = t["boxes"][:, 1] * h
                bw = t["boxes"][:, 2] * w
                bh = t["boxes"][:, 3] * h
                x_min = cx - bw / 2
                y_min = cy - bh / 2
                x_max = cx + bw / 2
                y_max = cy + bh / 2
                processed_targets.append({
                    "boxes": torch.stack([x_min, y_min, x_max, y_max], dim=1),
                    "labels": t["labels"]
                })
            map_metric.update(preds, processed_targets)
        else:
            map_metric.update(preds, targets)

    result = map_metric.compute()
    mAP = result["map_50"].item() if result["map_50"] is not None else 0.0
    mAR = result["mar_100"].item() if result["mar_100"] is not None else 0.0

    return avg_loss, mAP, mAR


def train_model(model, train_loader, val_loader, device, optimizer, scheduler, config):
    """
    Main training loop with warmup and optional detailed logging.
    Returns lists: train_losses, val_losses, map_list, mar_list.
    """
    num_epochs = config.get("num_epochs", 10)
    warmup_epochs = config.get("warmup_epochs", 0)
    verbose = config.get("verbose", False)
    train_losses, val_losses = [], []
    map_list, mar_list = [], []

    # Initialize warmup scheduler if needed
    warmup_scheduler = None
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs * len(train_loader)
        )

    for epoch in range(num_epochs):
        # Optionally log learning rate info at start of epoch
        if verbose:
            for i, group in enumerate(optimizer.param_groups):
                tqdm.write(f"[Epoch {epoch+1}] Param group {i}: lr = {group['lr']:.2e}, #params = {len(group['params'])}")

        if epoch == 0:
            num_trainable = sum(p.requires_grad for p in model.parameters())
            num_total = len(list(model.parameters()))
            tqdm.write(f"Initial trainable parameters: {num_trainable}/{num_total}")

        # Unfreeze backbone at specified freeze_epochs
        if config.get("freeze_backbone", False) and epoch == config.get("freeze_epochs", 0):
            tqdm.write("**** Unfreezing backbone parameters now ****")
            unfrozen_count = 0
            for name, param in model.backbone.named_parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    unfrozen_count += 1
            tqdm.write(f"Unfroze {unfrozen_count} backbone parameters (including BN layers).")
            backbone_params = list(model.backbone.parameters())
            existing_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
            new_backbone_params = [p for p in backbone_params if id(p) not in existing_param_ids]
            if new_backbone_params:
                optimizer.add_param_group({"params": new_backbone_params, "lr": config.get("backbone_lr", 1e-5)})
                tqdm.write(f"Added {len(new_backbone_params)} new backbone parameters to the optimizer.")
            else:
                tqdm.write("No new backbone parameters to add to the optimizer.")
            # Update head parameters' learning rate in groups not containing backbone parameters.
            for group in optimizer.param_groups:
                if not any(id(p) in {id(bp) for bp in backbone_params} for p in group["params"]):
                    group["lr"] = config.get("head_lr", 5e-3)
                    tqdm.write(f"Set head lr to {group['lr']:.2e} in one optimizer group.")
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                old_patience = config.get("plateau_patience", 5)
                old_factor = config.get("plateau_factor", 0.5)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=old_factor, patience=old_patience
                )
                tqdm.write("Reinitialized ReduceLROnPlateau scheduler with the updated optimizer.")

        model.train()
        running_loss = 0.0

        for batch_idx, (images, targets) in enumerate(tqdm(train_loader,
                                                           desc=f"Epoch {epoch+1}/{num_epochs}",
                                                           leave=False)):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss_val = compute_total_loss(loss_dict)
            running_loss += loss_val.item()

            # Detailed logging every 50 batches if verbose
            if verbose and (batch_idx % 50 == 0):
                loss_details = ", ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
                tqdm.write(f"[Epoch {epoch+1} Batch {batch_idx+1}] Loss: {loss_details}")

                backbone_grad_norm = 0.0
                head_grad_norm = 0.0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        norm = param.grad.data.norm().item()
                        if "backbone" in name:
                            backbone_grad_norm += norm
                        else:
                            head_grad_norm += norm
                tqdm.write(f"[Epoch {epoch+1} Batch {batch_idx+1}] Grad Norms: Backbone {backbone_grad_norm:.4e}, Head {head_grad_norm:.4e}")

            optimizer.zero_grad()
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            if warmup_scheduler and epoch < warmup_epochs:
                warmup_scheduler.step()

        avg_train_loss = running_loss / len(train_loader)
        val_loss, mAP, mAR = evaluate_on_dataset(model, val_loader, device, verbose=verbose)
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        map_list.append(mAP)
        mar_list.append(mAR)

        # Update scheduler
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                if epoch >= warmup_epochs:
                    scheduler.step()

        # Log epoch summary (always printed)
        lr_info = ", ".join([f"Group {i}: {group['lr']:.2e}" for i, group in enumerate(optimizer.param_groups)])
        tqdm.write(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                   f"mAP@0.5: {mAP:.4f} | mAR@100: {mAR:.4f} | LR: {lr_info}")

    return train_losses, val_losses, map_list, mar_list


