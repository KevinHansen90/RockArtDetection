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
        self.score_thresh = 0.1

    def forward(self, images, targets=None, orig_sizes=None):
        if isinstance(images, (list, tuple)):
            pixel_values = torch.stack(images)
            # Use provided original sizes if available, otherwise fall back to padded sizes.
            if orig_sizes is not None:
                target_sizes = [torch.tensor(s, device=img.device) for s, img in zip(orig_sizes, images)]
            else:
                target_sizes = [torch.tensor(img.shape[1:], device=img.device) for img in images]
        elif isinstance(images, torch.Tensor):
            pixel_values = images
            if orig_sizes is not None:
                target_sizes = [torch.tensor(s, device=pixel_values.device) for s in orig_sizes]
            else:
                target_sizes = [torch.tensor(pixel_values.shape[2:], device=pixel_values.device)] * pixel_values.shape[
                    0]
        else:
            raise TypeError(f"Unsupported input type for images: {type(images)}")

        # --- Training Mode ---
        if targets is not None:
            hf_labels = []
            for t in targets:
                hf_labels.append({
                    "class_labels": t["labels"],
                    "boxes": t["boxes"]
                })

            outputs = self.hf_model(pixel_values=pixel_values, labels=hf_labels)

            if hasattr(outputs, 'loss_dict'):
                return outputs.loss_dict
            elif hasattr(outputs, 'loss'):
                return {'total_loss': outputs.loss}
            else:
                print("Warning: Unexpected loss output format from Hugging Face model.")
                return {'total_loss': torch.tensor(0.0, device=pixel_values.device)}

        # --- Inference Mode ---
        else:
            outputs = self.hf_model(pixel_values=pixel_values)

            results = []
            for i, (logits_per_image, pred_boxes_per_image) in enumerate(zip(
                    outputs.logits, outputs.pred_boxes)):

                # Get the target size for this image (original [height, width])
                target_size = target_sizes[i]

                # Convert logits to probabilities
                probs = torch.softmax(logits_per_image, dim=-1)

                # For DETR, background is typically the last class.
                # Get scores and labels for the actual object classes.
                object_probs = probs[:, :-1]  # Exclude background
                scores, labels = object_probs.max(-1)

                # Filter out low-confidence predictions based on score threshold.
                keep = scores > self.score_thresh
                boxes = pred_boxes_per_image[keep]
                scores = scores[keep]
                labels = labels[keep]

                # Scale boxes to absolute image coordinates.
                # target_size is a tensor [orig_h, orig_w], so we use these directly.
                scale_x = target_size[1].item()  # Image width
                scale_y = target_size[0].item()  # Image height

                scaled_boxes = torch.zeros_like(boxes)
                scaled_boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * scale_x
                scaled_boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * scale_y
                scaled_boxes[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * scale_x
                scaled_boxes[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * scale_y

                scaled_boxes[:, 0] = torch.clamp(scaled_boxes[:, 0], min=0)
                scaled_boxes[:, 1] = torch.clamp(scaled_boxes[:, 1], min=0)
                scaled_boxes[:, 2] = torch.clamp(scaled_boxes[:, 2], max=target_size[1].item())
                scaled_boxes[:, 3] = torch.clamp(scaled_boxes[:, 3], max=target_size[0].item())

                results.append({
                    "scores": scores,
                    "labels": labels,
                    "boxes": scaled_boxes
                })

            return results


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
        # Load processor and model
        image_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr", use_fast=True)
        hf_model = DeformableDetrForObjectDetection.from_pretrained(
            "SenseTime/deformable-detr",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

        # Check if we have access to model.model which contains the actual architecture
        if hasattr(hf_model, "model"):
            # This is likely the correct structure - check if class_embed is a ModuleList
            if hasattr(hf_model.model, "class_embed") and isinstance(hf_model.model.class_embed, nn.ModuleList):
                # For each classification head in the list
                for i in range(len(hf_model.model.class_embed)):
                    in_features = hf_model.model.class_embed[i].in_features
                    hf_model.model.class_embed[i] = nn.Linear(in_features, num_classes + 1)

            # If it's a direct attribute
            elif hasattr(hf_model.model, "class_embed"):
                in_features = hf_model.model.class_embed.in_features
                hf_model.model.class_embed = nn.Linear(in_features, num_classes + 1)

        # Direct access to class_embed
        elif hasattr(hf_model, "class_embed"):
            if isinstance(hf_model.class_embed, nn.ModuleList):
                for i in range(len(hf_model.class_embed)):
                    in_features = hf_model.class_embed[i].in_features
                    hf_model.class_embed[i] = nn.Linear(in_features, num_classes + 1)
            else:
                in_features = hf_model.class_embed.in_features
                hf_model.class_embed = nn.Linear(in_features, num_classes + 1)

        # Fallback to d_model if defined in config
        else:
            if hasattr(hf_model.config, "d_model"):
                in_features = hf_model.config.d_model

                # Try to find where to place the classifier
                if hasattr(hf_model, "class_labels_classifier"):
                    hf_model.class_labels_classifier = nn.Linear(in_features, num_classes + 1)
                elif hasattr(hf_model, "model") and hasattr(hf_model.model, "class_labels_classifier"):
                    hf_model.model.class_labels_classifier = nn.Linear(in_features, num_classes + 1)
                else:
                    print("WARNING: Could not locate classification head to replace")
            else:
                print("WARNING: Could not determine input features dimension")

        # Make sure the model knows we're using a smaller number of classes
        hf_model.config.num_labels = num_classes

        # Wrap the configured HF model with your custom wrapper
        model = DeformableDETRWrapper(hf_model, image_processor)
        return model

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
