#!/usr/bin/env python3
import torch
from torch.backends import cudnn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection import MeanAveragePrecision
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm

from src.training.utils import compute_total_loss
from src.datasets.yolo_dataset import load_classes
from src.models.detection_models import DeformableDETRWrapper

# Speed optimization for fixed-size inputs
cudnn.benchmark = True

# One global metric instance with per-class breakdown
_map_metric = MeanAveragePrecision(class_metrics=True)


def evaluate_on_dataset(
    model,
    data_loader,
    device,
    classes: list,
    writer: SummaryWriter = None,
    step: int = None,
):
    """
    1) Computes avg validation loss.
    2) Computes COCO metrics (mAP@[.50:.95], mAP@.50, mAP@.75,
       mAR@1,10,100, by small/med/large).
    3) Computes simple F1 = 2·(P·R)/(P+R) at 50/100.
    4) Logs summary to console & to TensorBoard if writer/step given.
    Returns:
      val_loss, mAP50, mAR100, F1,
      mAP@[.50:.95], mAP@.75, mAR@1, mAR@10,
      mAP_small/med/large, mAR_small/med/large
    """
    # ——— 1) Validation‐loss pass ———
    model.train()
    total_loss = 0.0
    with torch.inference_mode():
        for imgs, tgts in data_loader:
            imgs = [i.to(device, non_blocking=True) for i in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            loss_dict = model(imgs, tgts)
            total_loss += compute_total_loss(loss_dict).item()
    val_loss = total_loss / len(data_loader)

    # ——— 2) Metric pass ———
    model.eval()
    _map_metric.reset()
    with torch.inference_mode():
        for imgs, tgts in data_loader:
            imgs = [i.to(device, non_blocking=True) for i in imgs]
            tgts = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in tgts]

            # Deformable DETR needs orig_sizes
            if isinstance(model, DeformableDETRWrapper):
                orig = [t["orig_size"].tolist() for t in tgts]
                preds = model(imgs, orig_sizes=orig)
            else:
                preds = model(imgs)

            # If DETR, convert normalized → absolute boxes
            if isinstance(model, DeformableDETRWrapper):
                processed = []
                for t in tgts:
                    b = t["boxes"]
                    if b.ndim == 2 and b.size(0) > 0:
                        h, w = t["orig_size"].tolist()
                        absb = torch.zeros_like(b)
                        absb[:, 0] = (b[:, 0] - b[:, 2] / 2) * w
                        absb[:, 1] = (b[:, 1] - b[:, 3] / 2) * h
                        absb[:, 2] = (b[:, 0] + b[:, 2] / 2) * w
                        absb[:, 3] = (b[:, 1] + b[:, 3] / 2) * h
                    else:
                        absb = b
                    processed.append({"boxes": absb, "labels": t["labels"]})
                _map_metric.update(preds, processed)
            else:
                _map_metric.update(preds, tgts)

    result = _map_metric.compute()

    # ——— Extract all metrics ———
    mAP    = result["map"].item()
    mAP50  = result["map_50"].item()
    mAP75  = result["map_75"].item()
    mAR1   = result["mar_1"].item()
    mAR10  = result["mar_10"].item()
    mAR100 = result["mar_100"].item()
    mAP_s  = result["map_small"].item()
    mAP_m  = result["map_medium"].item()
    mAP_l  = result["map_large"].item()
    mAR_s  = result["mar_small"].item()
    mAR_m  = result["mar_medium"].item()
    mAR_l  = result["mar_large"].item()

    # Simple F1 at 50/100
    f1 = 2 * mAP50 * mAR100 / (mAP50 + mAR100) if (mAP50 + mAR100) > 0 else 0.0

    # ——— Console summary ———
    tqdm.write(
        f"[Val] Loss: {val_loss:.4f} | mAP@.50: {mAP50:.3f} | mAR@100: {mAR100:.3f} | F1: {f1:.3f}"
    )

    # ——— TensorBoard logging ———
    if writer and step is not None:
        # core
        writer.add_scalar("Val/Loss",     val_loss, step)
        writer.add_scalar("Val/mAP@.50",  mAP50,    step)
        writer.add_scalar("Val/mAR@100", mAR100,   step)
        writer.add_scalar("Val/F1",      f1,       step)
        # full COCO suite
        writer.add_scalar("Val/mAP@[.50:.95]", mAP,    step)
        writer.add_scalar("Val/mAP@.75",      mAP75,  step)
        writer.add_scalar("Val/mAR@1",        mAR1,   step)
        writer.add_scalar("Val/mAR@10",       mAR10,  step)
        # by size
        writer.add_scalar("Val/mAP_small",    mAP_s,  step)
        writer.add_scalar("Val/mAP_medium",   mAP_m,  step)
        writer.add_scalar("Val/mAP_large",    mAP_l,  step)
        writer.add_scalar("Val/mAR_small",    mAR_s,  step)
        writer.add_scalar("Val/mAR_medium",   mAR_m,  step)
        writer.add_scalar("Val/mAR_large",    mAR_l,  step)

    return (
        val_loss, mAP50, mAR100, f1,
        mAP, mAP75, mAR1, mAR10,
        mAP_s, mAP_m, mAP_l,
        mAR_s, mAR_m, mAR_l
    )


def train_model(model, train_loader, val_loader, device, optimizer, scheduler, config):
    """
    Main training loop.
    Returns 15 lists in the order:
      train_losses,
      val_losses,
      map50s, mar100s, f1s,
      maps, map75s, mar1s, mar10s,
      map_small/med/large, mar_small/med/large
    """
    classes = load_classes(config["classes_file"])
    num_epochs    = config.get("num_epochs", 10)
    warmup_epochs = config.get("warmup_epochs", 0)
    accum_steps = max(1, config.get("grad_accum_steps", 1))

    writer = SummaryWriter(config["log_dir"]) if config.get("log_dir") else None

    # AMP scaler: only needed on CUDA
    scaler = GradScaler() if next(model.parameters()).device.type == "cuda" else None
    global_step = 0

    # Warmup scheduler
    warmup_scheduler = None
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs * len(train_loader),
        )

    # Initialize all metric lists
    train_losses = []
    val_losses   = []
    map50s = []; mar100s = []; f1s = []
    maps   = []; map75s = []
    mar1s  = []; mar10s = []
    m_s = []; m_m = []; m_l = []
    r_s = []; r_m = []; r_l = []

    for epoch in range(num_epochs):
        # ——— Training pass ———
        model.train()
        running_loss = 0.0
        accum_count = 0
        for imgs, tgts in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs = [i.to(device, non_blocking=True) for i in imgs]
            tgts = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in tgts]

            if accum_count == 0:
                optimizer.zero_grad(set_to_none=True)

            if scaler:
                # mixed‐precision
                with autocast():
                    loss_dict = model(imgs, tgts)
                    loss_val = compute_total_loss(loss_dict)
                scaler.scale(loss_val / accum_steps).backward()
                accum_count += 1
                if accum_count == accum_steps:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    scaler.step(optimizer)
                    scaler.update()
                    accum_count = 0
            else:
                # full FP32
                loss_dict = model(imgs, tgts)
                loss_val = compute_total_loss(loss_dict)
                (loss_val / accum_steps).backward()
                accum_count += 1
                if accum_count == accum_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()
                    accum_count = 0

            if warmup_scheduler and epoch < warmup_epochs:
                warmup_scheduler.step()

            running_loss += loss_val.item()
            if writer:
                writer.add_scalar("Train/Loss", loss_val.item(), global_step)
            global_step += 1

        if accum_count:
            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            accum_count = 0

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ——— Validation + metrics ———
        (
            val_loss, mAP50, mAR100, f1,
            mAP, mAP75, mAR1, mAR10,
            mAP_s, mAP_m, mAP_l,
            mAR_s, mAR_m, mAR_l
        ) = evaluate_on_dataset(
            model, val_loader, device, classes,
            writer=writer, step=global_step
        )
        val_losses.append(val_loss)
        map50s.append(mAP50)
        mar100s.append(mAR100)
        f1s.append(f1)
        maps.append(mAP)
        map75s.append(mAP75)
        mar1s.append(mAR1)
        mar10s.append(mAR10)
        m_s.append(mAP_s); m_m.append(mAP_m); m_l.append(mAP_l)
        r_s.append(mAR_s); r_m.append(mAR_m); r_l.append(mAR_l)

        # Scheduler step
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                if epoch >= warmup_epochs:
                    scheduler.step()

    if writer:
        writer.close()

    return (
        train_losses,
        val_losses,
        map50s, mar100s, f1s,
        maps, map75s, mar1s, mar10s,
        m_s, m_m, m_l,
        r_s, r_m, r_l,
    )
