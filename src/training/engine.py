##!/usr/bin/env python3
from __future__ import annotations

import time
from typing import List

import torch
from torch.backends import cudnn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from src.training.utils import (
    compute_total_loss,
    auto_amp_supported,
    get_cfg_dict,
)
from src.datasets.yolo_dataset import load_classes

try:
    from src.models.detection_models import DeformableDETRWrapper
except Exception:
    class _Dummy:
        ...
    DeformableDETRWrapper = _Dummy

cudnn.benchmark = True
_map_metric = MeanAveragePrecision(class_metrics=True)


# ────────────────────────────
# Gradient-accumulation helper
# ────────────────────────────
class GradAccumulator:
    """
    Accumulates micro-batches & performs an optimizer step every *every* calls.
    Returns *True* when an optimizer step has just been executed.
    """

    def __init__(self, model, optimizer, scaler: GradScaler | None, every: int):
        self.model, self.opt, self.scaler = model, optimizer, scaler
        self.every = max(1, every)
        self.count = 0

    # ---------------------------------------------------------------
    def zero(self):
        self.opt.zero_grad(set_to_none=True)

    def _step(self):
        if self.scaler and self.scaler.is_enabled():
            self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
        if self.scaler and self.scaler.is_enabled():
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.step()

    # ---------------------------------------------------------------
    def backward(self, loss: torch.Tensor) -> bool:
        """Scale & accumulate; return *True* when opt.step() was called."""
        if self.count == 0:
            self.zero()

        if self.scaler and self.scaler.is_enabled():
            self.scaler.scale(loss / self.every).backward()
        else:
            (loss / self.every).backward()

        self.count += 1
        stepped = False
        if self.count == self.every:
            self._step()
            self.count = 0
            stepped = True
        return stepped

    def flush(self):
        if self.count:
            self._step()
            self.count = 0


# ─────────────────────────────
# Validation helper (loss + mAP)
# ─────────────────────────────
def evaluate_on_dataset(
    model,
    data_loader,
    device,
    classes: List[str],
    writer: SummaryWriter | None = None,
    step: int | None = None,
):
    # ── loss pass ────────────────────────────────────────────────────
    model.train()
    total_loss = 0.0
    with torch.inference_mode():
        for imgs, tgts in data_loader:
            imgs = [i.to(device, non_blocking=True) for i in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            total_loss += compute_total_loss(model(imgs, tgts)).item()
    val_loss = total_loss / max(1, len(data_loader))

    # ── metric pass ──────────────────────────────────────────────────
    model.eval()
    _map_metric.reset()
    with torch.inference_mode():
        for imgs, tgts in data_loader:
            imgs = [i.to(device, non_blocking=True) for i in imgs]
            tgts = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in tgts]

            if isinstance(model, DeformableDETRWrapper):
                orig = [t["orig_size"].tolist() for t in tgts]
                preds = model(imgs, orig_sizes=orig)
                processed = []
                for t in tgts:
                    b = t["boxes"]
                    h, w = t["orig_size"].tolist()
                    if b.numel():
                        cx, cy, ww, hh = b.T
                        x1, y1 = (cx - ww / 2) * w, (cy - hh / 2) * h
                        x2, y2 = (cx + ww / 2) * w, (cy + hh / 2) * h
                        processed.append(
                            {"boxes": torch.stack([x1, y1, x2, y2], 1), "labels": t["labels"]}
                        )
                    else:
                        processed.append({"boxes": b, "labels": t["labels"]})
                _map_metric.update(preds, processed)
            else:
                _map_metric.update(model(imgs), tgts)

    res = _map_metric.compute()
    mAP50, mAR100 = res["map_50"].item(), res["mar_100"].item()
    f1 = 2 * mAP50 * mAR100 / (mAP50 + mAR100) if (mAP50 + mAR100) else 0.0
    tqdm.write(f"[Val] Loss {val_loss:.4f} | mAP@.50 {mAP50:.3f} | mAR@100 {mAR100:.3f} | F1 {f1:.3f}")

    if writer and step is not None:
        writer.add_scalar("Val/Loss", val_loss, step)
        writer.add_scalar("Val/mAP@.50", mAP50, step)
        writer.add_scalar("Val/mAR@100", mAR100, step)
        writer.add_scalar("Val/F1", f1, step)

    return (
        val_loss,
        mAP50,
        mAR100,
        f1,
        res["map"].item(),
        res["map_75"].item(),
        res["mar_1"].item(),
        res["mar_10"].item(),
        res["map_small"].item(),
        res["map_medium"].item(),
        res["map_large"].item(),
        res["mar_small"].item(),
        res["mar_medium"].item(),
        res["mar_large"].item(),
    )


# ─────────────────
# Training routine
# ─────────────────
def train_model(model, train_loader, val_loader, device, optimizer, scheduler, config, wandb_run=None):
    config = get_cfg_dict(config)

    # ─── freeze-N-epochs settings ──────────────────────────────────────
    freeze_epochs = config.get("freeze_epochs", 0)  # e.g. 2
    # keep a handle on backbone parameters for (un)freezing
    bb_params = list(model.backbone.parameters()) if hasattr(model, "backbone") else []
    # remember the original backbone LR so we can restore it later
    init_backbone_lr = optimizer.param_groups[0]["lr"]

    classes = load_classes(config["classes_file"])
    num_epochs = config.get("num_epochs", 10)
    warmup_epochs = config.get("warmup_epochs", 0)
    accum_steps = max(1, config.get("grad_accum_steps", 1))

    writer = SummaryWriter(config["log_dir"]) if config.get("log_dir") else None

    scaler = GradScaler(enabled=auto_amp_supported(device))
    accum = GradAccumulator(model, optimizer, scaler, accum_steps)

    warmup_sched = (
        LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs * len(train_loader),
        )
        if warmup_epochs
        else None
    )

    # metric buffers
    train_losses, val_losses = [], []
    map50s, mar100s, f1s = [], [], []
    maps, map75s, mar1s, mar10s = [], [], [], []
    m_s, m_m, m_l, r_s, r_m, r_l = [], [], [], [], [], []
    lr_backbones, lr_heads, epoch_times = [], [], []

    global_step = 0
    for epoch in range(num_epochs):
        # ─── dynamic backbone freeze/unfreeze ──────────────────────
        if epoch < freeze_epochs:
            # backbone frozen ⇒ 0 LR & disable grads
            optimizer.param_groups[0]["lr"] = 0.0
            for p in bb_params:
                p.requires_grad = False
        elif epoch == freeze_epochs:
            # first epoch *after* freeze ⇒ restore LR & grads
            optimizer.param_groups[0]["lr"] = init_backbone_lr
            for p in bb_params:
                p.requires_grad = True
            # if you have a scheduler, reset its base LR for group-0
            if scheduler is not None and hasattr(scheduler, "base_lrs"):
                scheduler.base_lrs[0] = init_backbone_lr

        start_t = time.time()
        model.train()
        running_loss = 0.0

        for imgs, tgts in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            imgs = [i.to(device, non_blocking=True) for i in imgs]
            tgts = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in tgts]

            with autocast(enabled=scaler.is_enabled()):
                loss_val = compute_total_loss(model(imgs, tgts))

            stepped = accum.backward(loss_val)

            # advance warm-up schedule *only* after optimizer.step()
            # and *only* if the backbone is not currently frozen
            if (
                warmup_sched
                and epoch < warmup_epochs
                and stepped
                and epoch >= freeze_epochs
            ):
                warmup_sched.step()

            running_loss += loss_val.item()
            if writer:
                writer.add_scalar("Train/Loss", loss_val.item(), global_step)
            global_step += 1

        accum.flush()
        train_losses.append(running_loss / len(train_loader))

        metrics = evaluate_on_dataset(
            model, val_loader, device, classes, writer=writer, step=global_step
        )
        (
            v_loss,
            m50,
            mar100,
            f1,
            m_all,
            m75,
            mar1,
            mar10,
            ms,
            mm,
            ml,
            rs,
            rm,
            rl,
        ) = metrics

        val_losses.append(v_loss)
        map50s.append(m50)
        mar100s.append(mar100)
        f1s.append(f1)
        maps.append(m_all)
        map75s.append(m75)
        mar1s.append(mar1)
        mar10s.append(mar10)
        m_s.extend([ms])
        m_m.extend([mm])
        m_l.extend([ml])
        r_s.extend([rs])
        r_m.extend([rm])
        r_l.extend([rl])

        lr_backbone = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]
        lr_backbones.append(lr_backbone)
        lr_heads.append(lr_head)
        epoch_sec = time.time() - start_t
        epoch_times.append(epoch_sec)

        if writer:
            writer.add_scalar("LR/backbone", lr_backbone, global_step)
            writer.add_scalar("LR/head", lr_head, global_step)
            writer.add_scalar("Time/epoch_sec", epoch_sec, epoch)

        # ── live W&B logging ──────────────────────────────────────────
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_losses[-1],
                    "val_loss": v_loss,
                    "mAP@.50": m50,
                    "mAR@100": mar100,
                    "lr_backbone": lr_backbone,
                    "lr_head": lr_head,
                    "epoch_time_s": epoch_sec,
                },
                step=epoch + 1,
            )

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(v_loss)
            elif epoch >= warmup_epochs:
                scheduler.step()

    if writer:
        writer.close()

    return (
        train_losses,
        val_losses,
        map50s,
        mar100s,
        f1s,
        maps,
        map75s,
        mar1s,
        mar10s,
        m_s,
        m_m,
        m_l,
        r_s,
        r_m,
        r_l,
        lr_backbones,
        lr_heads,
        epoch_times,
    )
