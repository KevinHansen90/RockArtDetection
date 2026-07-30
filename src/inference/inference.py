#!/usr/bin/env python3
"""
Local and Cloud Inference Script for Object Detection Models.
Generates side-by-side Ground Truth vs Prediction comparison collages.
"""

import os
import sys
import time
import argparse
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.models.detection_models import get_detection_model, DeformableDETRWrapper
from src.datasets.yolo_dataset import load_classes, YOLODataset
from src.training.utils import get_device, get_simple_transform
from src.training.evaluate import _draw_boxes


def load_model_for_inference(model_path: str, model_type: str, num_classes: int, device: torch.device):
    """Loads model weights and sets model to evaluation mode."""
    mt = model_type.lower()
    num_classes_model = num_classes if mt == "deformable_detr" else num_classes + 1

    model = get_detection_model(mt, num_classes_model, config={})
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"Loaded model '{model_type}' from {model_path} onto {device}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Run inference and generate GT vs Prediction visual collages.")
    parser.add_argument("--model-path", required=True, help="Path to trained .pth checkpoint")
    parser.add_argument("--input", required=True, help="Input images directory")
    parser.add_argument("--labels", required=True, help="Ground truth labels directory")
    parser.add_argument("--output", required=True, help="Output directory for comparison collages")
    parser.add_argument("--classes", required=True, help="Path to grouped_classes.txt file")
    parser.add_argument("--model-type", required=True, choices=["fasterrcnn", "retinanet", "deformable_detr", "yolov5"], help="Model architecture")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", default=None, help="cuda | mps | cpu")

    args = parser.parse_args()

    device = get_device() if not args.device else torch.device(args.device)
    print(f"Using device: {device}")

    class_names = load_classes(args.classes)
    num_classes = len(class_names)

    model = load_model_for_inference(args.model_path, args.model_type, num_classes, device)
    mt = args.model_type.lower()

    os.makedirs(args.output, exist_ok=True)
    test_tf = get_simple_transform()

    test_dataset = YOLODataset(
        images_dir=args.input,
        labels_dir=args.labels,
        classes_file=args.classes,
        mode="test",
        transforms=test_tf,
        normalize_boxes=(mt == "deformable_detr"),
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0])
    print(f"Processing {len(test_dataset)} images for inference...")

    comparison_images = []
    with torch.no_grad():
        for i, (pil_img, img_tensor, gt_boxes, gt_labels) in enumerate(tqdm(test_loader, desc="Inference")):
            orig_w, orig_h = pil_img.size

            if mt == "deformable_detr":
                pred = model(img_tensor.unsqueeze(0).to(device), orig_sizes=[[orig_h, orig_w]])[0]
            else:
                pred = model(img_tensor.unsqueeze(0).to(device))[0]

            keep = pred["scores"].cpu() >= args.threshold
            pred_boxes = pred["boxes"].cpu()[keep].tolist()
            pred_labels = pred["labels"].cpu()[keep].tolist()

            if mt in {"fasterrcnn", "retinanet"}:
                pred_labels = [l - 1 if l > 0 else l for l in pred_labels]

            gt_boxes_list = gt_boxes.tolist() if isinstance(gt_boxes, torch.Tensor) else gt_boxes
            gt_labels_list = gt_labels.tolist() if isinstance(gt_labels, torch.Tensor) else gt_labels

            gt_vis = _draw_boxes(pil_img.copy(), gt_boxes_list, gt_labels_list, class_names, color="green")
            pred_vis = _draw_boxes(pil_img.copy(), pred_boxes, pred_labels, class_names, color="red")

            combined = Image.new("RGB", (orig_w * 2, orig_h))
            combined.paste(gt_vis, (0, 0))
            combined.paste(pred_vis, (orig_w, 0))

            comparison_images.append(combined)

    if comparison_images:
        total_h = sum(img.height for img in comparison_images)
        max_w = max(img.width for img in comparison_images)
        final_collage = Image.new("RGB", (max_w, total_h), color="white")

        y_offset = 0
        for img in comparison_images:
            final_collage.paste(img, (0, y_offset))
            y_offset += img.height

        out_path = os.path.join(args.output, f"inference_collage_{mt}_{int(time.time())}.png")
        final_collage.save(out_path)
        print(f"Saved inference collage -> {out_path}")


if __name__ == "__main__":
    main()
