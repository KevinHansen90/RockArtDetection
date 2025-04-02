# src/models/detection_models.py

import torch
import torch.nn as nn
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


# Import Hugging Face components for Deformable DETR
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection


class DeformableDETRWrapper(torch.nn.Module):
    def __init__(self, hf_model, image_processor):
        super().__init__()
        self.hf_model = hf_model
        self.image_processor = image_processor
        self.score_thresh = 0.5

    def forward(self, images, targets=None):
        # Convert list of tensors -> batched tensor
        pixel_values = torch.stack(images)  # (B, C, H, W)

        if targets is not None:
            # Training mode - return losses
            outputs = self.hf_model(pixel_values=pixel_values, labels=targets)
            return {"loss": outputs.loss}  # Match TorchVision format
        else:
            # Inference mode - return predictions
            outputs = self.hf_model(pixel_values=pixel_values)
            # Get image sizes for post-processing
            target_sizes = [img.shape[1:] for img in images]  # (H, W)
            # Standardized prediction format
            return self.image_processor.post_process_object_detection(
                outputs,
                threshold=self.score_thresh,
                target_sizes=target_sizes
            )


class CustomRetinaNetClassificationHead(RetinaNetClassificationHead):
    def __init__(self, in_channels, num_anchors, num_classes, norm_layer,
                 focal_loss_gamma=2.5, focal_loss_alpha=0.25, prior_probability=0.01):
        # Pass norm_layer and prior_probability as keywords.
        super().__init__(in_channels, num_anchors, num_classes,
                         norm_layer=norm_layer, prior_probability=prior_probability)
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha

        # Save the original function from the focal_loss module.
        self._original_sigmoid_focal_loss = focal_loss_module.sigmoid_focal_loss

        # Define a custom sigmoid_focal_loss that always uses your parameters.
        def custom_sigmoid_focal_loss(inputs, targets, reduction="sum", **kwargs):
            return self._original_sigmoid_focal_loss(inputs, targets, reduction=reduction,
                                                     gamma=self.focal_loss_gamma, alpha=self.focal_loss_alpha, **kwargs)
        # Monkey-patch the module's sigmoid_focal_loss.
        focal_loss_module.sigmoid_focal_loss = custom_sigmoid_focal_loss


def get_detection_model(model_type, num_classes, config=None):
    """
    Creates and returns a detection model with custom classifier heads for `num_classes`.
    For deformable_detr, note that Hugging Face's model expects:
      - `num_labels` = number of object classes (background is handled internally),
      - and targets with boxes normalized to [0, 1].
    """
    model_type = model_type.lower()

    if model_type == "fasterrcnn":
        # First create model with default anchors
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)

        # Now modify anchors
        anchor_sizes = tuple((x, int(x * 1.5), x * 2) for x in [128, 256, 512, 1024, 2048])  # 64, 128, 256, 512, 1024
        aspect_ratios = ((1.0, 3.0, 6.0),) * len(anchor_sizes)

        # Replace both RPN and head anchors
        model.rpn.anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )

        # Must also update RPN head to match new anchor count
        num_anchors = model.rpn.anchor_generator.num_anchors_per_location()[0]
        in_channels = model.backbone.out_channels  # Typically 256 for FPN

        model.rpn.head = RPNHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            conv_depth=1  # Keep original depth
        )

        # Then modify ROI head as before
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model

    elif model_type == "retinanet":
        weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
        model = retinanet_resnet50_fpn_v2(weights=weights)

        # Adjusted anchor generator: 9 anchors per location (3 scales x 3 aspect ratios)
        anchor_sizes = tuple((x, int(x * 1.5), x * 2) for x in [128, 256, 512, 1024, 2048])  #64, 128, 256, 512, 1024
        aspect_ratios = ((1.0, 3.0, 6.0),) * len(anchor_sizes)
        model.anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )

        focal_config = config.get("focal_loss", {})
        gamma = focal_config.get("gamma", 2.5)
        alpha = focal_config.get("alpha", 0.25)
        prior_probability = focal_config.get("prior_probability", 0.01)
        num_anchors = 9  # 3 scales x 3 ratios
        in_channels = 256
        model.head.classification_head = CustomRetinaNetClassificationHead(
            in_channels,
            num_anchors,
            num_classes,
            norm_layer=torch.nn.BatchNorm2d,
            focal_loss_gamma=gamma,
            focal_loss_alpha=alpha,
            prior_probability=prior_probability
        )

        return model

    elif model_type == "deformable_detr":
        image_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr", use_fast=True)
        hf_model = DeformableDetrForObjectDetection.from_pretrained(
            "SenseTime/deformable-detr",
            ignore_mismatched_sizes=True
        )

        # Set the number of object classes (excluding the no-object class)
        hf_model.config.num_labels = num_classes
        # Reinitialize classifier head: output dimension = num_classes + 1 (for no-object)
        hf_model.class_labels_classifier = nn.Linear(hf_model.config.d_model, num_classes + 1)
        hf_model.config.eos_coef = 0.1

        # Wrap tensor lists as ParameterList for query_embed, level_embed, refpoint_embed:
        for attr in ['query_embed', 'level_embed', 'refpoint_embed']:
            if hasattr(hf_model, attr):
                value = getattr(hf_model, attr)
                location = hf_model
            elif hasattr(hf_model, "model") and hasattr(hf_model.model, attr):
                value = getattr(hf_model.model, attr)
                location = hf_model.model
            else:
                continue  # Attribute not found; skip wrapping
            if isinstance(value, list):
                wrapped = nn.ParameterList([
                    p if isinstance(p, nn.Parameter) else nn.Parameter(torch.tensor(p))
                    for p in value
                ])
            elif isinstance(value, torch.Tensor):
                wrapped = nn.Parameter(value)
            else:
                raise TypeError(f"Unexpected type for {attr}: {type(value)}")
            setattr(location, attr, wrapped)
        # Now wrap the HF model with your custom wrapper.
        model = DeformableDETRWrapper(hf_model, image_processor)

        return model

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
