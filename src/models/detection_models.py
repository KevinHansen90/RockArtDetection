#!/usr/bin/env python3
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LRScheduler,
    MultiStepLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.rpn import RPNHead
import torchvision.ops.focal_loss as focal_loss_module

# Hugging Face imports for Deformable-DETR
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection

# ── our helpers -------------------------------------------------------------
from src.training.utils import get_cfg_dict  # ← unified cfg → dict helper

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 0.  Deformable-DETR wrapper                                                 #
# --------------------------------------------------------------------------- #
class DeformableDETRWrapper(nn.Module):
    """Unifies HF Deformable-DETR with TorchVision detector API."""

    def __init__(self, hf_model, image_processor, score_thresh: float = 0.1):
        super().__init__()
        self.hf_model = hf_model
        self.image_processor = image_processor
        self.score_thresh = score_thresh

    # --- forward ---------------------------------------------------------- #
    def forward(
        self,
        images: torch.Tensor | List[torch.Tensor],
        targets: Optional[List[dict]] = None,
        orig_sizes: Optional[List[List[int]]] = None,
    ):
        # Stack list into batch tensor if needed
        if isinstance(images, (list, tuple)):
            pixel_values = torch.stack(images)
            target_sizes = [
                torch.as_tensor(s, device=pixel_values.device)
                for s in (
                    orig_sizes
                    or [i.shape[-2:] for i in images]  # fallback to images' own size
                )
            ]
        else:
            pixel_values = images
            h, w = pixel_values.shape[-2:]
            target_sizes = [torch.tensor([h, w], device=pixel_values.device)] * pixel_values.size(0)

        # ---------------- Training path ----------------
        if targets is not None:
            hf_labels = [
                {"class_labels": t["labels"], "boxes": t["boxes"]} for t in targets
            ]
            out = self.hf_model(pixel_values=pixel_values, labels=hf_labels)
            return out.loss_dict if hasattr(out, "loss_dict") else {"total_loss": out.loss}

        # ---------------- Inference path ---------------
        outputs = self.hf_model(pixel_values=pixel_values)
        results = []
        for logits, boxes, size in zip(outputs.logits, outputs.pred_boxes, target_sizes):
            probs = torch.softmax(logits, dim=-1)
            scores, labels = probs[:, :-1].max(dim=-1)  # drop background column
            keep = scores > self.score_thresh
            sel_boxes = boxes[keep]
            h, w = size
            xyxy = torch.zeros_like(sel_boxes)
            xyxy[:, 0] = (sel_boxes[:, 0] - sel_boxes[:, 2] / 2) * w
            xyxy[:, 1] = (sel_boxes[:, 1] - sel_boxes[:, 3] / 2) * h
            xyxy[:, 2] = (sel_boxes[:, 0] + sel_boxes[:, 2] / 2) * w
            xyxy[:, 3] = (sel_boxes[:, 1] + sel_boxes[:, 3] / 2) * h
            results.append(
                {"boxes": xyxy, "scores": scores[keep], "labels": labels[keep]}
            )
        return results


# --------------------------------------------------------------------------- #
# 1.  Dataclass config                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class DetectorCfg:
    """Subset of YAML keys that control the model factory."""
    model_type: str

    # DETR-specific
    num_queries: int = 300

    # Anchor params (for RetinaNet & Faster-RCNN)
    anchor_sizes: Optional[List[List[int]]] = None  # per-feature level
    anchor_ratios: Optional[List[List[float]]] = None

    # Focal-loss params (RetinaNet)
    focal_gamma: float = 2.5
    focal_alpha: float = 0.25
    focal_prior: float = 0.01

    # Optimiser params
    freeze_backbone: bool = False
    backbone_lr: float = 5e-5
    head_lr: float = 5e-4
    optimizer: str = "adamw"
    weight_decay: float = 5e-4
    momentum: float = 0.9
    eps: float = 1e-7

    # --------------------------------------------------------------------- #
    @staticmethod
    def from_dict(d: Dict) -> "DetectorCfg":
        """Create cfg, flattening *anchor_params* / *focal_loss* groups."""
        a_cfg = d.get("anchor_params", {})
        f_cfg = d.get("focal_loss", {})
        flat = {
            **d,
            "anchor_sizes": a_cfg.get("sizes"),
            "anchor_ratios": a_cfg.get("ratios"),
            "focal_gamma": f_cfg.get("gamma", 2.5),
            "focal_alpha": f_cfg.get("alpha", 0.25),
            "focal_prior": f_cfg.get("prior_probability", 0.01),
        }
        allowed = DetectorCfg.__annotations__.keys()
        clean = {k: v for k, v in flat.items() if k in allowed}
        return DetectorCfg(**clean)


# --------------------------------------------------------------------------- #
# 2.  Internal helpers                                                        #
# --------------------------------------------------------------------------- #
def _make_anchor_generator(cfg: DetectorCfg) -> AnchorGenerator:
    """Return a TorchVision AnchorGenerator from cfg sizes/ratios."""
    if cfg.anchor_sizes:
        sizes = tuple(tuple(l) for l in cfg.anchor_sizes)
    else:  # COCO-like default (5 levels, 3 scales)
        sizes = (
            (32, 64, 128),
            (64, 128, 256),
            (128, 256, 512),
            (256, 512, 1024),
            (512, 1024, 2048),
        )
    if cfg.anchor_ratios:
        ratios = tuple(tuple(r) for r in cfg.anchor_ratios)
    else:
        ratios = ((0.5, 1.0, 2.0),) * len(sizes)
    return AnchorGenerator(sizes=sizes, aspect_ratios=ratios)


@contextmanager
def _patch_focal_loss(gamma: float, alpha: float, prior: float):
    """Temporarily patch TorchVision sigmoid_focal_loss with new params."""
    orig_fn = focal_loss_module.sigmoid_focal_loss

    def _custom(inputs, targets, **kw):
        return orig_fn(
            inputs,
            targets,
            gamma=gamma,
            alpha=alpha,
            reduction=kw.get("reduction", "sum"),
        )

    focal_loss_module.sigmoid_focal_loss = _custom  # type: ignore
    try:
        yield
    finally:
        focal_loss_module.sigmoid_focal_loss = orig_fn


class CustomRetinaNetClassificationHead(RetinaNetClassificationHead):
    """Wraps classification head to isolate focal-loss patch."""

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_classes: int,
        norm_layer,
        focal_loss_gamma: float,
        focal_loss_alpha: float,
        prior_probability: float,
    ):
        super().__init__(
            in_channels,
            num_anchors,
            num_classes,
            norm_layer=norm_layer,
            prior_probability=prior_probability,
        )
        self._orig_focal = focal_loss_module.sigmoid_focal_loss
        focal_loss_module.sigmoid_focal_loss = (
            lambda inputs, targets, **kw: self._orig_focal(
                inputs,
                targets,
                gamma=focal_loss_gamma,
                alpha=focal_loss_alpha,
                reduction=kw.get("reduction", "sum"),
            )
        )

    def __del__(self):
        focal_loss_module.sigmoid_focal_loss = self._orig_focal


# --------------------------------------------------------------------------- #
# 3.  Factory                                                                 #
# --------------------------------------------------------------------------- #
def get_detection_model(
    model_type: str | None,
    num_classes: int,
    config: Optional[Dict] = None,
) -> nn.Module:  # noqa: C901
    """
    Return a TorchVision or HF detection model ready for training.

    *config* may be a Hydra DictConfig, plain ``dict``, or ``DetectorCfg``.
    """
    if isinstance(config, DetectorCfg):
        cfg = config
    else:
        cfg = DetectorCfg.from_dict(get_cfg_dict(config or {}))

    mt = (model_type or cfg.model_type).lower()

    # -------------------- Faster R-CNN -----------------------------
    if mt == "fasterrcnn":
        model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        )
        model.rpn.anchor_generator = _make_anchor_generator(cfg)
        in_channels = model.backbone.out_channels
        n_anchors = model.rpn.anchor_generator.num_anchors_per_location()[0]
        model.rpn.head = RPNHead(in_channels, n_anchors)
        in_feat = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
        return model

    # -------------------- RetinaNet --------------------------------
    if mt == "retinanet":
        model = retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
        )
        model.anchor_generator = _make_anchor_generator(cfg)

        # Classification head with custom focal loss
        with _patch_focal_loss(cfg.focal_gamma, cfg.focal_alpha, cfg.focal_prior):
            num_anchors = model.anchor_generator.num_anchors_per_location()[0]
            model.head.classification_head = CustomRetinaNetClassificationHead(
                in_channels=256,
                num_anchors=num_anchors,
                num_classes=num_classes,
                norm_layer=torch.nn.BatchNorm2d,
                focal_loss_gamma=cfg.focal_gamma,
                focal_loss_alpha=cfg.focal_alpha,
                prior_probability=cfg.focal_prior,
            )
        model.num_classes = num_classes
        return model

    # -------------------- Deformable-DETR --------------------------
    if mt == "deformable_detr":
        processor = AutoImageProcessor.from_pretrained(
            "SenseTime/deformable-detr", use_fast=True
        )
        hf_model = DeformableDetrForObjectDetection.from_pretrained(
            "SenseTime/deformable-detr",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        hf_model.config.num_queries = cfg.num_queries
        return DeformableDETRWrapper(hf_model, processor)

    raise ValueError(f"Unknown model_type: {mt}")


# --------------------------------------------------------------------------- #
# 4.  Optimiser & scheduler helpers                                           #
# --------------------------------------------------------------------------- #
def get_optimizer(model: nn.Module, config: Dict) -> Optimizer:
    cfg = (
        config if isinstance(config, DetectorCfg) else DetectorCfg.from_dict(get_cfg_dict(config))
    )

    bb_params = (
        list(model.backbone.parameters()) if hasattr(model, "backbone") else []
    )
    bb_ids = set(map(id, bb_params))
    head_params = [p for p in model.parameters() if id(p) not in bb_ids]

    if cfg.freeze_backbone:
        for p in bb_params:
            p.requires_grad = False
        param_groups = [
            {"params": head_params, "lr": cfg.head_lr},
        ]
    else:
        param_groups = [
            {"params": bb_params, "lr": cfg.backbone_lr},
            {"params": head_params, "lr": cfg.head_lr},
        ]

    opt_type = cfg.optimizer.lower()
    if opt_type == "sgd":
        return optim.SGD(
            param_groups, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
    if opt_type == "adam":
        return optim.Adam(param_groups, weight_decay=cfg.weight_decay)
    if opt_type == "adamw":
        return optim.AdamW(
            param_groups, weight_decay=cfg.weight_decay, eps=cfg.eps
        )
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")


def get_scheduler(optimizer: Optimizer, config: Dict) -> Optional[LRScheduler]:
    cfg_dict = get_cfg_dict(config)
    sname = (cfg_dict.get("scheduler") or "none").lower()

    if sname == "none":
        return None
    if sname == "steplr":
        return StepLR(
            optimizer,
            step_size=cfg_dict.get("step_size", 7),
            gamma=cfg_dict.get("gamma", 0.1),
        )
    if sname == "reducelronplateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg_dict.get("plateau_factor", 0.5),
            patience=cfg_dict.get("plateau_patience", 5),
        )
    if sname == "cosineannealinglr":
        return CosineAnnealingLR(
            optimizer,
            T_max=cfg_dict.get("T_max", 10),
            eta_min=cfg_dict.get("eta_min", 0),
        )
    if sname == "cosineannealingwarmrestarts":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg_dict.get("T_0", 10),
            T_mult=cfg_dict.get("T_mult", 1),
            eta_min=cfg_dict.get("eta_min", 0),
        )
    if sname == "multisteplr":
        return MultiStepLR(
            optimizer,
            milestones=cfg_dict.get("milestones", [30, 60]),
            gamma=cfg_dict.get("gamma", 0.1),
        )
    if sname == "onecyclelr":
        return OneCycleLR(
            optimizer,
            max_lr=cfg_dict.get(
                "max_lr", [pg["lr"] for pg in optimizer.param_groups]
            ),
            total_steps=cfg_dict.get("total_steps", 100),
            pct_start=cfg_dict.get("pct_start", 0.3),
            anneal_strategy=cfg_dict.get("anneal_strategy", "cos"),
            cycle_momentum=cfg_dict.get("cycle_momentum", True),
            base_momentum=cfg_dict.get("base_momentum", 0.85),
            max_momentum=cfg_dict.get("max_momentum", 0.95),
            div_factor=cfg_dict.get("div_factor", 25.0),
            final_div_factor=cfg_dict.get("final_div_factor", 10000.0),
        )
    raise ValueError(f"Unsupported scheduler: {sname}")
