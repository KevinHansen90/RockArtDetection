#!/usr/bin/env python3
from __future__ import annotations
import logging, math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch, torch.nn as nn
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

from src.training.utils import get_cfg_dict  # ← your helper

log = logging.getLogger(__name__)

# ─────────────────────────── 0  De-DETR wrapper ─────────────────────────────
class DeformableDETRWrapper(nn.Module):
    def __init__(self, hf_model, image_processor, score_thresh: float = 0.1):
        super().__init__()
        self.hf_model, self.image_processor, self.score_thresh = \
            hf_model, image_processor, score_thresh

    def forward(self, images, targets: Optional[List[dict]] = None,
                orig_sizes: Optional[List[List[int]]] = None):
        if isinstance(images, (list, tuple)):
            pixel_values = torch.stack(images)
            target_sizes = [torch.as_tensor(s, device=pixel_values.device)
                            for s in (orig_sizes or [i.shape[-2:] for i in images])]
        else:
            pixel_values = images
            h, w = pixel_values.shape[-2:]
            target_sizes = [torch.tensor([h, w], device=pixel_values.device)] \
                           * pixel_values.size(0)

        if targets is not None:                       # training branch
            hf_labels = [{"class_labels": t["labels"], "boxes": t["boxes"]}
                         for t in targets]
            out = self.hf_model(pixel_values=pixel_values, labels=hf_labels)
            return out.loss_dict if hasattr(out, "loss_dict") else {"loss": out.loss}

        # inference branch
        outs = self.hf_model(pixel_values=pixel_values)
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

# ─────────────────────────── 1  Cfg dataclass ───────────────────────────────
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
    def from_dict(d: Dict) -> "DetectorCfg":
        a, f = d.get("anchor_params", {}), d.get("focal_loss", {})
        flat = {**d,
                "anchor_sizes": a.get("sizes"), "anchor_ratios": a.get("ratios"),
                "focal_gamma": f.get("gamma", 2.5),
                "focal_alpha": f.get("alpha", 0.25),
                "focal_prior": f.get("prior_probability", 0.01)}
        return DetectorCfg(**{k: flat[k] for k in flat if k in DetectorCfg.__annotations__})

# ─────────────────────────── 2  helpers ─────────────────────────────────────
def _make_anchor_generator(cfg: DetectorCfg) -> AnchorGenerator:
    if cfg.anchor_sizes:
        sizes = tuple(tuple(s) for s in cfg.anchor_sizes)
    else:
        sizes = ((32, 64, 128), (64, 128, 256),
                 (128, 256, 512), (256, 512, 1024), (512, 1024, 2048))
    ratios = tuple(tuple(r) for r in
                   (cfg.anchor_ratios or ((0.5, 1.0, 2.0),) * len(sizes)))
    return AnchorGenerator(sizes, ratios)

@contextmanager
def _patch_focal(gamma: float, alpha: float):
    orig = focal_loss_module.sigmoid_focal_loss
    focal_loss_module.sigmoid_focal_loss = \
        lambda i, t, **kw: orig(i, t, gamma=gamma, alpha=alpha,
                                reduction=kw.get("reduction", "sum"))
    try:  yield
    finally: focal_loss_module.sigmoid_focal_loss = orig

def _copy_class_tower(src: nn.Module, dst: nn.Module):
    src_dict = src.state_dict()
    # keep every tensor that is *not* the logits layer
    filtered = {k: v for k, v in src_dict.items() if "cls_logits" not in k}
    missing, unexpected = dst.load_state_dict(filtered, strict=False)
    assert not unexpected  # should be empty; logits is the only filtered key
    # `missing` now only contains dst.cls_logits.* and is expected

def _build_retinanet_head(num_anchors: int, num_classes: int, prior: float):
    new_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=nn.BatchNorm2d,
        prior_probability=prior,
    )
    # proper bias initialisation for focal loss
    bias_val = -math.log((1 - prior) / prior)
    nn.init.constant_(new_head.cls_logits.bias, bias_val)
    return new_head

# ─────────────────────────── 3  Model factory ───────────────────────────────
def get_detection_model(model_type: str | None, num_classes: int,
                        config: Optional[Dict] = None) -> nn.Module:
    cfg = config if isinstance(config, DetectorCfg) else \
          DetectorCfg.from_dict(get_cfg_dict(config or {}))
    mt = (model_type or cfg.model_type).lower()

    # ---------- Faster R-CNN -------------------------------------------------
    if mt == "fasterrcnn":
        m = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
        m.rpn.anchor_generator = _make_anchor_generator(cfg)

        # rebuild RPN head for new anchor count
        in_ch = m.backbone.out_channels
        n_anch = m.rpn.anchor_generator.num_anchors_per_location()[0]
        m.rpn.head = RPNHead(in_ch, n_anch)

        # keep shared fc layers, replace only logits/bbox rows
        in_feat = m.roi_heads.box_predictor.cls_score.in_features
        new_pred = FastRCNNPredictor(in_feat, num_classes)
        # copy weights except last dim mismatch
        with torch.no_grad():
            old_pred = m.roi_heads.box_predictor
            # fc8 weights: (N_classes_old, 1024) -> take top rows
            new_pred.cls_score.weight.copy_(old_pred.cls_score.weight[:num_classes])
            new_pred.cls_score.bias.copy_(old_pred.cls_score.bias[:num_classes])
            new_pred.bbox_pred.weight.copy_(old_pred.bbox_pred.weight[
                :num_classes * 4])
            new_pred.bbox_pred.bias.copy_(old_pred.bbox_pred.bias[
                :num_classes * 4])
        m.roi_heads.box_predictor = new_pred
        return m

    # ---------- RetinaNet ----------------------------------------------------
    if mt == "retinanet":
        m = retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
        )
        m.anchor_generator = _make_anchor_generator(cfg)
        num_anchors = m.anchor_generator.num_anchors_per_location()[0]

        old_head = m.head.classification_head
        new_head = _build_retinanet_head(num_anchors, num_classes, cfg.focal_prior)

        _copy_class_tower(old_head, new_head)

        with _patch_focal(cfg.focal_gamma, cfg.focal_alpha):
            m.head.classification_head = new_head
        m.num_classes = num_classes
        return m

    # ---------- Deformable-DETR ---------------------------------------------
    if mt == "deformable_detr":
        proc = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
        hf = DeformableDetrForObjectDetection.from_pretrained(
            "SenseTime/deformable-detr",
            num_labels=num_classes, ignore_mismatched_sizes=True)
        hf.config.num_queries = cfg.num_queries
        return DeformableDETRWrapper(hf, proc)

    raise ValueError(f"Unknown model_type {mt}")

# ─────────────────────────── 4  Optimiser / scheduler ───────────────────────
def get_optimizer(model: nn.Module, config: Dict) -> Optimizer:
    cfg = config if isinstance(config, DetectorCfg) else \
          DetectorCfg.from_dict(get_cfg_dict(config))
    bb_params = list(model.backbone.parameters()) if hasattr(model, "backbone") else []
    bb_ids = {id(p) for p in bb_params}
    hd_params = [p for p in model.parameters() if id(p) not in bb_ids]

    groups = [
        {"params": bb_params, "lr": cfg.backbone_lr},  # group-0  backbone
        {"params": hd_params, "lr": cfg.head_lr},  # group-1  heads
        ]

    o = cfg.optimizer.lower()
    if o == "sgd":   return optim.SGD(groups, momentum=cfg.momentum,
                                      weight_decay=cfg.weight_decay)
    if o == "adam":  return optim.Adam(groups, weight_decay=cfg.weight_decay)
    if o == "adamw": return optim.AdamW(groups, weight_decay=cfg.weight_decay,
                                        eps=cfg.eps)
    raise ValueError(f"Unknown optimizer {cfg.optimizer}")

def get_scheduler(opt: Optimizer, cfg_dict: Dict) -> Optional[LRScheduler]:
    cfg = get_cfg_dict(cfg_dict)
    name = (cfg.get("scheduler") or "none").lower()
    if name == "none": return None
    if name == "steplr":
        return StepLR(opt, step_size=cfg.get("step_size", 7),
                      gamma=cfg.get("gamma", 0.1))
    if name == "reducelronplateau":
        return ReduceLROnPlateau(opt, mode="min",
                                 factor=cfg.get("plateau_factor", 0.5),
                                 patience=cfg.get("plateau_patience", 5))
    if name == "cosineannealinglr":
        return CosineAnnealingLR(opt, T_max=cfg.get("T_max", 10),
                                 eta_min=cfg.get("eta_min", 0))
    if name == "cosineannealingwarmrestarts":
        return CosineAnnealingWarmRestarts(opt,
                                           T_0=cfg.get("T_0", 10),
                                           T_mult=cfg.get("T_mult", 1),
                                           eta_min=cfg.get("eta_min", 0))
    if name == "multisteplr":
        return MultiStepLR(opt,
                           milestones=cfg.get("milestones", [30, 60]),
                           gamma=cfg.get("gamma", 0.1))
    if name == "onecyclelr":
        return OneCycleLR(opt, max_lr=cfg.get("max_lr",
                                              [pg["lr"] for pg in opt.param_groups]),
                          total_steps=cfg.get("total_steps", 100),
                          pct_start=cfg.get("pct_start", 0.3),
                          anneal_strategy=cfg.get("anneal_strategy", "cos"),
                          cycle_momentum=cfg.get("cycle_momentum", True),
                          base_momentum=cfg.get("base_momentum", 0.85),
                          max_momentum=cfg.get("max_momentum", 0.95),
                          div_factor=cfg.get("div_factor", 25.0),
                          final_div_factor=cfg.get("final_div_factor", 10000.0))
    raise ValueError(f"Unknown scheduler {name}")
