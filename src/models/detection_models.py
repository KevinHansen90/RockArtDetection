#!/usr/bin/env python3

import logging
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LRScheduler,
    ReduceLROnPlateau,
    StepLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    MultiStepLR,
    OneCycleLR
)
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
import torchvision.ops.focal_loss as focal_loss_module
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead

# Hugging Face imports
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection

# Initialize module logger
torch_logger = logging.getLogger(__name__)
torch_logger.setLevel(logging.INFO)


class DeformableDETRWrapper(nn.Module):
    """
    Wraps a Hugging Face Deformable DETR model for training and inference.
    """
    def __init__(self, hf_model, image_processor):
        super().__init__()
        self.hf_model = hf_model
        self.image_processor = image_processor
        self.score_thresh = 0.1

    def forward(self, images, targets=None, orig_sizes: Optional[List[List[int]]] = None):
        # Prepare pixel values
        if isinstance(images, (list, tuple)):
            pixel_values = torch.stack(images)
            target_sizes = [torch.tensor(s, device=img.device)
                            for s, img in zip(orig_sizes or [], images)]
        elif isinstance(images, torch.Tensor):
            pixel_values = images
            bs = pixel_values.shape[0]
            target_sizes = [torch.tensor(s, device=pixel_values.device)
                            for s in (orig_sizes or [pixel_values.shape[2:]] * bs)]
        else:
            raise TypeError(f"Unsupported images type: {type(images)}")

        # Training mode: return loss dict
        if targets is not None:
            hf_labels = [{"class_labels": t["labels"], "boxes": t["boxes"]} for t in targets]
            outputs = self.hf_model(pixel_values=pixel_values, labels=hf_labels)
            if hasattr(outputs, 'loss_dict'):
                return outputs.loss_dict
            return {"total_loss": outputs.loss}

        # Inference mode: post-process outputs
        outputs = self.hf_model(pixel_values=pixel_values)
        results = []
        for logits, boxes, size in zip(outputs.logits, outputs.pred_boxes, target_sizes):
            probs = torch.softmax(logits, dim=-1)
            scores, labels = probs[:, :-1].max(-1)  # exclude background
            keep = scores > self.score_thresh
            sel_boxes = boxes[keep]
            # scale to absolute coords
            h, w = size.tolist()
            xyxy = torch.zeros_like(sel_boxes)
            xyxy[:, 0] = (sel_boxes[:, 0] - sel_boxes[:, 2]/2) * w
            xyxy[:, 1] = (sel_boxes[:, 1] - sel_boxes[:, 3]/2) * h
            xyxy[:, 2] = (sel_boxes[:, 0] + sel_boxes[:, 2]/2) * w
            xyxy[:, 3] = (sel_boxes[:, 1] + sel_boxes[:, 3]/2) * h
            results.append({"scores": scores[keep], "labels": labels[keep], "boxes": xyxy})
        return results


class CustomRetinaNetClassificationHead(RetinaNetClassificationHead):
    """
    RetinaNet classification head with custom focal loss parameters.
    """
    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_classes: int,
        norm_layer,
        focal_loss_gamma: float = 2.5,
        focal_loss_alpha: float = 0.25,
        prior_probability: float = 0.01
    ):
        super().__init__(in_channels, num_anchors, num_classes,
                         norm_layer=norm_layer,
                         prior_probability=prior_probability)
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha
        self._orig_focal = focal_loss_module.sigmoid_focal_loss
        focal_loss_module.sigmoid_focal_loss = lambda inputs, targets, **kwargs: \
            self._orig_focal(inputs, targets,
                             gamma=self.focal_loss_gamma,
                             alpha=self.focal_loss_alpha,
                             reduction=kwargs.get("reduction", "sum"))


def get_detection_model(model_type: str, num_classes: int, config: Optional[Dict] = None) -> nn.Module:
    """
    Factory for detection models: 'fasterrcnn', 'retinanet', 'deformable_detr'.
    """
    mt = model_type.lower()
    if mt == "fasterrcnn":
        model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
        sizes = tuple((x, int(x*1.5), x*2) for x in [128,256,512,1024,2048])
        ratios = ((1.0,3.0,6.0),) * len(sizes)
        model.rpn.anchor_generator = AnchorGenerator(sizes, ratios)
        in_channels = model.backbone.out_channels
        n_anchors = model.rpn.anchor_generator.num_anchors_per_location()[0]
        model.rpn.head = RPNHead(in_channels, n_anchors)
        in_feat = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
        return model
    elif mt == "retinanet":
        weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
        model = retinanet_resnet50_fpn_v2(weights=weights)
        anchor_sizes = tuple((x, int(x * 1.5), x * 2)
                             for x in [128, 256, 512, 1024, 2048])
        aspect_ratios = ((1.0, 3.0, 6.0),) * len(anchor_sizes)
        model.anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )
        focal_cfg = config.get("focal_loss", {})
        gamma = focal_cfg.get("gamma", 2.5)
        alpha = focal_cfg.get("alpha", 0.25)
        prior_prob = focal_cfg.get("prior_probability", 0.01)
        num_anchors = model.anchor_generator.num_anchors_per_location()[0]
        in_channels = 256
        model.head.classification_head = CustomRetinaNetClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=torch.nn.BatchNorm2d,
            focal_loss_gamma=gamma,
            focal_loss_alpha=alpha,
            prior_probability=prior_prob
        )
        model.num_classes = num_classes
        return model
    elif mt == "deformable_detr":
        processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr", use_fast=True)
        hf_model = DeformableDetrForObjectDetection.from_pretrained(
            "SenseTime/deformable-detr",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        return DeformableDETRWrapper(hf_model, processor)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def get_optimizer(model: nn.Module, config: Dict) -> Optimizer:
    freeze_bb = config.get("freeze_backbone", False)

    # Collect backbone parameters (if any)
    bb_params = list(model.backbone.parameters()) if hasattr(model, 'backbone') else []

    # If freezing backbone, only optimize the rest
    if freeze_bb:
        for p in bb_params:
            p.requires_grad = False
        trainable = [p for p in model.parameters() if p.requires_grad]
        groups = [
            {"params": trainable, "lr": config.get("head_lr", 5e-4)}
        ]
        torch_logger.info(f"Backbone frozen; optimizing {len(trainable)} params.")
    else:
        # Partition model parameters into backbone vs head by identity
        bb_ids = set(map(id, bb_params))
        head_params = [p for p in model.parameters() if id(p) not in bb_ids]
        groups = [
            {"params": bb_params,   "lr": config.get("backbone_lr", 5e-5)},
            {"params": head_params, "lr": config.get("head_lr",    5e-4)}
        ]
        torch_logger.info(
            f"Optimizing backbone ({len(bb_params)}) + head ({len(head_params)}) parameters."
        )

    # Construct optimizer
    opt_type = config.get("optimizer", "adamw").lower()
    wd = config.get("weight_decay", 5e-4)
    if opt_type == "sgd":
        return optim.SGD(groups,
                         momentum=config.get("momentum", 0.9),
                         weight_decay=wd)
    if opt_type == "adam":
        return optim.Adam(groups,
                          weight_decay=wd)
    if opt_type == "adamw":
        return optim.AdamW(groups,
                           weight_decay=wd,
                           eps=config.get("eps", 1e-7))
    raise ValueError(f"Unsupported optimizer: {opt_type}")


def get_scheduler(optimizer: Optimizer, config: Dict) -> Optional[LRScheduler]:
    sname = config.get("scheduler")
    if not sname or sname.lower() == "none":
        return None
    sname = sname.lower()
    torch_logger.info(f"Setting up scheduler: {sname}")
    if sname == "steplr":
        return StepLR(optimizer, step_size=config.get("step_size",7), gamma=config.get("gamma",0.1))
    if sname == "reducelronplateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=config.get("plateau_factor",0.5), patience=config.get("plateau_patience",5))
    if sname == "cosineannealinglr":
        return CosineAnnealingLR(optimizer, T_max=config.get("T_max",10), eta_min=config.get("eta_min",0))
    if sname == "cosineannealingwarmrestarts":
        return CosineAnnealingWarmRestarts(optimizer, T_0=config.get("T_0",10), T_mult=config.get("T_mult",1), eta_min=config.get("eta_min",0))
    if sname == "multisteplr":
        return MultiStepLR(optimizer, milestones=config.get("milestones",[30,60]), gamma=config.get("gamma",0.1))
    if sname == "onecyclelr":
        return OneCycleLR(optimizer, max_lr=config.get("max_lr",[pg['lr'] for pg in optimizer.param_groups]), total_steps=config.get("total_steps",100), pct_start=config.get("pct_start",0.3), anneal_strategy=config.get("anneal_strategy","cos"), cycle_momentum=config.get("cycle_momentum",True), base_momentum=config.get("base_momentum",0.85), max_momentum=config.get("max_momentum",0.95), div_factor=config.get("div_factor",25.0), final_div_factor=config.get("final_div_factor",10000.0))
    raise ValueError(f"Unsupported scheduler: {sname}")
