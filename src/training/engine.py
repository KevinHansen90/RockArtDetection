#!/usr/bin/env python3
from __future__ import annotations

import torch
from torch.backends import cudnn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from src.training.utils import compute_total_loss, auto_amp_supported
from src.datasets.yolo_dataset import load_classes

# optional import – avoids hard crash if wrapper moves again
try:
    from src.models.detection_models import DeformableDETRWrapper  # type: ignore
except Exception:  # pragma: no cover
    class _Dummy: ...
    DeformableDETRWrapper = _Dummy  # type: ignore

cudnn.benchmark = True
_map_metric = MeanAveragePrecision(class_metrics=True)

# --------------------------------------------------------------------------- #
# Gradient accumulator                                                        #
# --------------------------------------------------------------------------- #
class GradAccumulator:
    """Context-free helper to handle micro-batch accumulation."""

    def __init__(self, model, optimizer, scaler: GradScaler | None, every: int):
        self.model = model
        self.opt = optimizer
        self.scaler = scaler
        self.every = max(1, every)
        self.count = 0

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

    # ---- public --------------------------------------------------------- #
    def backward(self, loss: torch.Tensor):
        if self.count == 0:
            self.zero()

        if self.scaler and self.scaler.is_enabled():
            self.scaler.scale(loss / self.every).backward()
        else:
            (loss / self.every).backward()

        self.count += 1
        if self.count == self.every:
            self._step()
            self.count = 0

    def flush(self):
        if self.count:  # handle leftovers at epoch end
            self._step()
            self.count = 0


# --------------------------------------------------------------------------- #
# Val / metrics helper (unchanged logic)                                      #
# --------------------------------------------------------------------------- #
def evaluate_on_dataset(
    model,
    data_loader,
    device,
    classes: list[str],
    writer: SummaryWriter | None = None,
    step: int | None = None,
):
    # ----- loss pass ---------------------------------------------------- #
    model.train()
    total_loss = 0.0
    with torch.inference_mode():
        for imgs, tgts in data_loader:
            imgs = [i.to(device, non_blocking=True) for i in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            total_loss += compute_total_loss(model(imgs, tgts)).item()
    val_loss = total_loss / max(1, len(data_loader))

    # ----- metric pass -------------------------------------------------- #
    model.eval(); _map_metric.reset()
    with torch.inference_mode():
        for imgs, tgts in data_loader:
            imgs = [i.to(device, non_blocking=True) for i in imgs]
            tgts = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in tgts]

            if isinstance(model, DeformableDETRWrapper):
                orig = [t["orig_size"].tolist() for t in tgts]
                preds = model(imgs, orig_sizes=orig)
                # convert GT boxes to xyxy
                processed = []
                for t in tgts:
                    b = t["boxes"]; h, w = t["orig_size"].tolist()
                    if b.numel():
                        cx, cy, ww, hh = b.T
                        x1 = (cx - ww / 2) * w
                        y1 = (cy - hh / 2) * h
                        x2 = (cx + ww / 2) * w
                        y2 = (cy + hh / 2) * h
                        processed.append({"boxes": torch.stack([x1, y1, x2, y2], 1), "labels": t["labels"]})
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
        val_loss, mAP50, mAR100, f1,
        res["map"].item(), res["map_75"].item(),
        res["mar_1"].item(), res["mar_10"].item(),
        res["map_small"].item(),  res["map_medium"].item(),  res["map_large"].item(),
        res["mar_small"].item(),  res["mar_medium"].item(),  res["mar_large"].item(),
    )


# --------------------------------------------------------------------------- #
# Training loop                                                               #
# --------------------------------------------------------------------------- #
def train_model(model, train_loader, val_loader, device, optimizer, scheduler, config):
    classes = load_classes(config["classes_file"])
    num_epochs    = config.get("num_epochs", 10)
    warmup_epochs = config.get("warmup_epochs", 0)
    accum_steps   = max(1, config.get("grad_accum_steps", 1))

    writer = SummaryWriter(config["log_dir"]) if config.get("log_dir") else None

    # AMP & scaler (CUDA or MPS)
    scaler = GradScaler(enabled=auto_amp_supported(device))
    accum  = GradAccumulator(model, optimizer, scaler, accum_steps)

    if warmup_epochs:
        warmup_sched = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=warmup_epochs * len(train_loader),
        )
    else:
        warmup_sched = None

    # metric buffers
    train_losses, val_losses = [], []
    map50s, mar100s, f1s     = [], [], []
    maps, map75s             = [], []
    mar1s, mar10s            = [], []
    m_s, m_m, m_l            = [], [], []
    r_s, r_m, r_l            = [], [], []

    global_step = 0
    for epoch in range(num_epochs):
        model.train(); running_loss = 0.0

        for imgs, tgts in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs = [i.to(device, non_blocking=True) for i in imgs]
            tgts = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in tgts]

            with autocast(enabled=scaler.is_enabled()):
                loss_dict = model(imgs, tgts)
                loss_val  = compute_total_loss(loss_dict)

            accum.backward(loss_val)

            if warmup_sched and epoch < warmup_epochs:
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
            v_loss, m50, mar100, f1,
            m_all, m75, mar1, mar10,
            ms, mm, ml, rs, rm, rl,
        ) = metrics

        val_losses.append(v_loss)
        map50s.append(m50); mar100s.append(mar100); f1s.append(f1)
        maps.append(m_all);  map75s.append(m75)
        mar1s.append(mar1);  mar10s.append(mar10)
        m_s.append(ms);      m_m.append(mm);    m_l.append(ml)
        r_s.append(rs);      r_m.append(rm);    r_l.append(rl)

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(v_loss)
            elif epoch >= warmup_epochs:
                scheduler.step()

    if writer:
        writer.close()

    return (
        train_losses, val_losses,
        map50s, mar100s, f1s,
        maps, map75s, mar1s, mar10s,
        m_s, m_m, m_l,
        r_s, r_m, r_l,
    )
