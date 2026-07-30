#!/usr/bin/env python3
"""
Unified Object Detection Model Zoo
Supports: Faster R-CNN, RetinaNet, Deformable DETR, and Ultralytics YOLO (v5/v8/v11).
"""
from __future__ import annotations
import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, CosineAnnealingWarmRestarts, MultiStepLR,
    OneCycleLR, ReduceLROnPlateau, StepLR, LRScheduler,
)

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2, retinanet_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights, RetinaNet_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.rpn import RPNHead
import torchvision.ops.focal_loss as focal_loss_module

# Hugging Face Deformable-DETR
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection

# Ultralytics YOLO
try:
    from ultralytics import YOLO
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.cfg import get_cfg
except ImportError:
    YOLO = None
    v8DetectionLoss = None
    get_cfg = None

from src.training.utils import get_cfg_dict

log = logging.getLogger(__name__)


# ─────────────────────────── De-DETR wrapper ─────────────────────────────
class DeformableDETRWrapper(nn.Module):
    def __init__(self, hf_model, image_processor, score_thresh: float = 0.001):
        super().__init__()
        self.hf_model = hf_model
        self.image_processor = image_processor
        self.score_thresh = score_thresh

    def forward(self, images, targets: Optional[List[dict]] = None,
                orig_sizes: Optional[List[List[int]]] = None):
        if isinstance(images, (list, tuple)):
            pixel_values = torch.stack(images)
            target_sizes = [torch.as_tensor(s, device=pixel_values.device)
                            for s in (orig_sizes or [i.shape[-2:] for i in images])]
        else:
            pixel_values = images
            h, w = pixel_values.shape[-2:]
            target_sizes = [torch.tensor([h, w], device=pixel_values.device)] * pixel_values.size(0)

        if targets is not None:
            hf_labels = []
            for t in targets:
                lbls = t["labels"].long() if t["labels"].dtype != torch.long else t["labels"]
                bxs = t["boxes"].float()
                hf_labels.append({"class_labels": lbls, "boxes": bxs})
            out = self.hf_model(pixel_values=pixel_values, labels=hf_labels)
            return out.loss_dict if hasattr(out, "loss_dict") else {"loss": out.loss}

        outs = self.hf_model(pixel_values=pixel_values)
        if self.image_processor is not None and hasattr(self.image_processor, "post_process_object_detection"):
            target_sizes_tensor = torch.stack(target_sizes)
            dets = self.image_processor.post_process_object_detection(
                outs, target_sizes=target_sizes_tensor, threshold=self.score_thresh
            )
            return dets

        dets = []
        for logits, boxes, size in zip(outs.logits, outs.pred_boxes, target_sizes):
            probs = torch.softmax(logits, dim=-1)
            scores, labels = probs[:, :-1].max(dim=-1)
            keep = scores > self.score_thresh
            sel_boxes = boxes[keep]
            h, w = size
            xyxy = torch.zeros_like(sel_boxes)
            xyxy[:, 0] = (sel_boxes[:, 0] - sel_boxes[:, 2] / 2) * w
            xyxy[:, 1] = (sel_boxes[:, 1] - sel_boxes[:, 3] / 2) * h
            xyxy[:, 2] = (sel_boxes[:, 0] + sel_boxes[:, 2] / 2) * w
            xyxy[:, 3] = (sel_boxes[:, 1] + sel_boxes[:, 3] / 2) * h
            dets.append({"boxes": xyxy, "scores": scores[keep], "labels": labels[keep]})
        return dets


# ─────────────────────────── Ultralytics YOLO wrapper ────────────────────
class YOLOWrapper(nn.Module):
    """
    Unified PyTorch wrapper for Ultralytics YOLO (v5/v8/v11).
    Allows running YOLO directly in PyTorch training engine with real loss computation.
    """
    def __init__(self, model_name: str = "yolov5su.pt", num_classes: int = 2):
        super().__init__()
        if YOLO is None:
            raise ImportError("ultralytics package is required for YOLO models. Install with pip install ultralytics")
        self._yolo = YOLO(model_name)
        self.model = self._yolo.model
        self.num_classes = num_classes
        self.model.nc = num_classes
        head = self.model.model[-1]
        head.nc = num_classes
        if hasattr(head, "cv3"):
            for i in range(len(head.cv3)):
                in_c = head.cv3[i][2].in_channels
                head.cv3[i][2] = nn.Conv2d(in_c, num_classes, kernel_size=1, stride=1)
        if get_cfg is not None:
            self.model.args = get_cfg()
            self.loss_fn = v8DetectionLoss(self.model)
        else:
            self.loss_fn = None

    def train(self, mode: bool = True):
        self.training = mode
        self.model.train(mode)
        return self

    def forward(self, images, targets: Optional[List[dict]] = None):
        if isinstance(images, (list, tuple)):
            max_h = max(img.shape[1] for img in images)
            max_w = max(img.shape[2] for img in images)
            pad_h = (32 - (max_h % 32)) % 32
            pad_w = (32 - (max_w % 32)) % 32
            target_h, target_w = max_h + pad_h, max_w + pad_w
            padded = [
                torch.nn.functional.pad(img, (0, target_w - img.shape[2], 0, target_h - img.shape[1]))
                for img in images
            ]
            pixel_values = torch.stack(padded)
        else:
            h, w = images.shape[-2:]
            pad_h = (32 - (h % 32)) % 32
            pad_w = (32 - (w % 32)) % 32
            if pad_h > 0 or pad_w > 0:
                pixel_values = torch.nn.functional.pad(images, (0, pad_w, 0, pad_h))
            else:
                pixel_values = images

        if self.training and targets is not None:
            dev = pixel_values.device
            batch_idxs, cls_list, box_list = [], [], []
            for i, t in enumerate(targets):
                b = t["boxes"]
                lbls = t["labels"].float()
                if b.numel():
                    h, w = pixel_values.shape[-2:]
                    x1, y1, x2, y2 = b.T
                    cx = (x1 + x2) / 2.0 / w
                    cy = (y1 + y2) / 2.0 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    norm_boxes = torch.stack([cx, cy, bw, bh], dim=1)
                    batch_idxs.append(torch.full((b.size(0),), i, device=dev, dtype=torch.float32))
                    cls_list.append(lbls)
                    box_list.append(norm_boxes)
            if batch_idxs:
                target_dict = {
                    "batch_idx": torch.cat(batch_idxs),
                    "cls": torch.cat(cls_list),
                    "bboxes": torch.cat(box_list),
                }
            else:
                target_dict = {
                    "batch_idx": torch.empty((0,), device=dev),
                    "cls": torch.empty((0,), device=dev),
                    "bboxes": torch.empty((0, 4), device=dev),
                }
            preds = self.model(pixel_values)
            if self.loss_fn is not None:
                self.loss_fn.device = dev
                if hasattr(self.loss_fn, "proj") and isinstance(self.loss_fn.proj, torch.Tensor):
                    self.loss_fn.proj = self.loss_fn.proj.to(dev)
                train_preds = preds[1] if isinstance(preds, (tuple, list)) and len(preds) > 1 else preds
                loss_components, _ = self.loss_fn(train_preds, target_dict)
                loss_box = loss_components[0]
                loss_cls = loss_components[1]
                loss_dfl = loss_components[2] if len(loss_components) > 2 else torch.tensor(0.0, device=dev)
                dummy = 0.0 * sum(p.sum() for p in self.model.parameters() if p.requires_grad)
                return {
                    "loss_box": loss_box + dummy,
                    "loss_cls": loss_cls + dummy,
                    "loss_dfl": loss_dfl + dummy,
                }
        else:
            dets = []
            dev = pixel_values.device
            h_img, w_img = pixel_values.shape[-2:]
            raw_preds = self.model(pixel_values)
            raw_outs = raw_preds[0] if isinstance(raw_preds, (tuple, list)) else raw_preds
            
            import torchvision
            for i in range(raw_outs.shape[0]):
                out = raw_outs[i] # (4 + num_classes, num_anchors)
                boxes_cxcywh = out[:4, :].T
                cls_probs = out[4:, :].T
                
                scores, labels = cls_probs.max(dim=1)
                keep = scores > 0.01
                
                if not keep.any():
                    dets.append({
                        "boxes": torch.empty((0, 4), device=dev),
                        "scores": torch.empty((0,), device=dev),
                        "labels": torch.empty((0,), dtype=torch.long, device=dev),
                    })
                    continue
                    
                b_keep = boxes_cxcywh[keep]
                s_keep = scores[keep]
                l_keep = labels[keep]
                
                cx, cy, bw, bh = b_keep.T
                x1 = torch.clamp((cx - bw / 2), 0, w_img)
                y1 = torch.clamp((cy - bh / 2), 0, h_img)
                x2 = torch.clamp((cx + bw / 2), 0, w_img)
                y2 = torch.clamp((cy + bh / 2), 0, h_img)
                xyxy = torch.stack([x1, y1, x2, y2], dim=1)
                
                nms_keep = torchvision.ops.nms(xyxy, s_keep, 0.45)
                dets.append({
                    "boxes": xyxy[nms_keep],
                    "scores": s_keep[nms_keep],
                    "labels": l_keep[nms_keep] + 1,
                })
            return dets


# ─────────────────────────── Cfg dataclass ───────────────────────────────
@dataclass
class DetectorCfg:
    model_type: str
    num_queries: int = 300
    anchor_sizes: Optional[List[List[int]]] = None
    anchor_ratios: Optional[List[List[float]]] = None
    focal_gamma: float = 2.5
    focal_alpha: float = 0.25
    focal_prior: float = 0.01
    backbone_lr: float = 5e-5
    head_lr: float = 5e-4
    optimizer: str = "adamw"
    weight_decay: float = 5e-4
    momentum: float = 0.9
    eps: float = 1e-7

    @staticmethod
    def from_dict(d: Dict) -> DetectorCfg:
        a, f = d.get("anchor_params", {}), d.get("focal_loss", {})
        flat = {
            **d,
            "anchor_sizes": a.get("sizes"), "anchor_ratios": a.get("ratios"),
            "focal_gamma": f.get("gamma", 2.5),
            "focal_alpha": f.get("alpha", 0.25),
            "focal_prior": f.get("prior_probability", 0.01),
        }
        return DetectorCfg(**{k: flat[k] for k in flat if k in DetectorCfg.__annotations__})


def _make_anchor_generator(cfg: DetectorCfg) -> AnchorGenerator:
    if cfg.anchor_sizes:
        raw = cfg.anchor_sizes
        if len(raw) == 5:
            sizes = tuple(tuple(s) if isinstance(s, (list, tuple)) else (s,) for s in raw)
        elif len(raw) == 1:
            elem = tuple(raw[0]) if isinstance(raw[0], (list, tuple)) else (raw[0],)
            sizes = (elem,) * 5
        else:
            sizes = tuple(tuple(s) if isinstance(s, (list, tuple)) else (s,) for s in raw)
    else:
        sizes = ((32,), (64,), (128,), (256,), (512,))
    ratios = tuple(tuple(r) if isinstance(r, (list, tuple)) else (r,) for r in (cfg.anchor_ratios or ((0.5, 1.0, 2.0),) * len(sizes)))
    if len(ratios) != len(sizes):
        ratios = (ratios[0],) * len(sizes)
    return AnchorGenerator(sizes, ratios)


@contextmanager
def _patch_focal(gamma: float, alpha: float):
    orig = focal_loss_module.sigmoid_focal_loss
    focal_loss_module.sigmoid_focal_loss = lambda i, t, **kw: orig(i, t, gamma=gamma, alpha=alpha, reduction=kw.get("reduction", "sum"))
    try:
        yield
    finally:
        focal_loss_module.sigmoid_focal_loss = orig


def _copy_class_tower(src: nn.Module, dst: nn.Module):
    src_dict = src.state_dict()
    filtered = {k: v for k, v in src_dict.items() if "cls_logits" not in k}
    dst.load_state_dict(filtered, strict=False)


def _build_retinanet_head(num_anchors: int, num_classes: int, prior: float):
    new_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=nn.BatchNorm2d,
        prior_probability=prior,
    )
    bias_val = -math.log((1 - prior) / prior)
    nn.init.constant_(new_head.cls_logits.bias, bias_val)
    return new_head


# ─────────────────────────── Model factory ───────────────────────────────
def get_detection_model(model_type: str | None, num_classes: int, config: Optional[Dict] = None) -> nn.Module:
    cfg = config if isinstance(config, DetectorCfg) else DetectorCfg.from_dict(get_cfg_dict(config or {}))
    mt = (model_type or cfg.model_type).lower()

    if mt == "fasterrcnn":
        m = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
        if cfg.anchor_sizes is not None:
            m.rpn.anchor_generator = _make_anchor_generator(cfg)
            in_ch = m.backbone.out_channels
            n_anch = m.rpn.anchor_generator.num_anchors_per_location()[0]
            m.rpn.head = RPNHead(in_ch, n_anch)

        in_feat = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
        m.roi_heads.score_thresh = 0.01
        return m

    if mt == "retinanet":
        m = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)
        if cfg.anchor_sizes is not None:
            m.anchor_generator = _make_anchor_generator(cfg)

        num_anchors = m.anchor_generator.num_anchors_per_location()[0]
        old_head = m.head.classification_head
        new_head = _build_retinanet_head(num_anchors, num_classes, cfg.focal_prior)

        _copy_class_tower(old_head, new_head)

        with _patch_focal(cfg.focal_gamma, cfg.focal_alpha):
            m.head.classification_head = new_head
        m.num_classes = num_classes
        m.score_thresh = 0.01
        return m


    if mt == "deformable_detr":
        proc = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
        hf = DeformableDetrForObjectDetection.from_pretrained(
            "SenseTime/deformable-detr",
            num_labels=num_classes, ignore_mismatched_sizes=True
        )
        hf.config.num_queries = cfg.num_queries
        return DeformableDETRWrapper(hf, proc)

    if mt.startswith("yolo"):
        weights_name = f"{mt}.pt" if not mt.endswith(".pt") else mt
        if weights_name == "yolov5.pt":
            weights_name = "yolov5s.pt"
        return YOLOWrapper(weights_name, num_classes)

    raise ValueError(f"Unknown model_type {mt}")


# ─────────────────────────── Optimizer / scheduler ───────────────────────
def get_optimizer(model: nn.Module, config: Dict) -> Optimizer:
    cfg = config if isinstance(config, DetectorCfg) else DetectorCfg.from_dict(get_cfg_dict(config))
    bb_params = list(model.backbone.parameters()) if hasattr(model, "backbone") else []
    bb_ids = {id(p) for p in bb_params}
    hd_params = [p for p in model.parameters() if id(p) not in bb_ids]

    groups = [
        {"params": bb_params, "lr": cfg.backbone_lr},
        {"params": hd_params, "lr": cfg.head_lr},
    ]

    o = cfg.optimizer.lower()
    if o == "sgd":
        return optim.SGD(groups, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    if o == "adam":
        return optim.Adam(groups, weight_decay=cfg.weight_decay)
    if o == "adamw":
        return optim.AdamW(groups, weight_decay=cfg.weight_decay, eps=cfg.eps)
    raise ValueError(f"Unknown optimizer {cfg.optimizer}")


def get_scheduler(opt: Optimizer, cfg_dict: Dict) -> Optional[LRScheduler]:
    cfg = get_cfg_dict(cfg_dict)
    name = (cfg.get("scheduler") or "none").lower()
    if name == "none":
        return None
    if name == "steplr":
        return StepLR(opt, step_size=cfg.get("step_size", 7), gamma=cfg.get("gamma", 0.1))
    if name == "reducelronplateau":
        return ReduceLROnPlateau(opt, mode="min", factor=cfg.get("plateau_factor", 0.5), patience=cfg.get("plateau_patience", 5))
    if name == "cosineannealinglr":
        return CosineAnnealingLR(opt, T_max=cfg.get("T_max", 10), eta_min=cfg.get("eta_min", 0))
    if name == "cosineannealingwarmrestarts":
        return CosineAnnealingWarmRestarts(opt, T_0=cfg.get("T_0", 10), T_mult=cfg.get("T_mult", 1), eta_min=cfg.get("eta_min", 0))
    if name == "multisteplr":
        return MultiStepLR(opt, milestones=cfg.get("milestones", [30, 60]), gamma=cfg.get("gamma", 0.1))
    if name == "onecyclelr":
        return OneCycleLR(
            opt,
            max_lr=cfg.get("max_lr", [pg["lr"] for pg in opt.param_groups]),
            total_steps=cfg.get("total_steps", 100),
            pct_start=cfg.get("pct_start", 0.3),
            anneal_strategy=cfg.get("anneal_strategy", "cos"),
            cycle_momentum=cfg.get("cycle_momentum", True),
            base_momentum=cfg.get("base_momentum", 0.85),
            max_momentum=cfg.get("max_momentum", 0.95),
            div_factor=cfg.get("div_factor", 25.0),
            final_div_factor=cfg.get("final_div_factor", 10000.0),
        )
    raise ValueError(f"Unknown scheduler {name}")
