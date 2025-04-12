#!/usr/bin/env python3

import argparse
import os
import sys
import time
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from torch.utils.data import DataLoader

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_SCRIPT_DIR)
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.detection_models import get_detection_model, DeformableDETRWrapper
from src.datasets.yolo_dataset import load_classes, TestDataset
from src.training.utils import get_device, get_simple_transform
from src.training.evaluate import draw_boxes as evaluate_draw_boxes


def load_model_for_inference(model_path, model_type, num_classes_actual, device):
    """Loads model architecture, state dict, sets to eval mode."""
    model_type = model_type.lower()
    if model_type == "deformable_detr":
        num_classes_model = num_classes_actual
    else:
        num_classes_model = num_classes_actual + 1
    model = get_detection_model(model_type, num_classes_model, config={})
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
             state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        print(f"Successfully loaded model weights from {model_path}")
    except Exception as e:
        print(f"Error loading model weights from {model_path}: {e}")
        raise
    model.to(device)
    model.eval()
    print(f"Model '{model_type}' loaded on device '{device}' and set to evaluation mode.")
    return model


def preprocess_image(pil_image, device, model_type, image_processor=None):
    """Preprocesses a PIL image."""
    model_type = model_type.lower()
    if model_type == "deformable_detr":
        if image_processor is None:
             raise ValueError("Image processor must be provided for Deformable DETR.")
        inputs = image_processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        return pixel_values # Return only the tensor needed for model input
    else:
        transform = get_simple_transform()
        image_tensor = transform(pil_image).to(device)
        return image_tensor.unsqueeze(0) # Add batch dimension


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference and compare with ground truth.")
    parser.add_argument("--model-path", required=True, help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("--input", required=True, help="Path to directory of input images.")
    parser.add_argument("--labels", required=True, help="Path to directory of corresponding ground truth label files (.txt).") # Added
    parser.add_argument("--output", required=True, help="Directory to save output comparison images.")
    parser.add_argument("--classes", required=True, help="Path to the file containing class names (one per line).")
    parser.add_argument("--model-type", required=True, choices=["fasterrcnn", "retinanet", "deformable_detr"], help="Type of model architecture used.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for displaying detections.")
    parser.add_argument("--device", default=None, help="Device to use ('cuda', 'cpu', 'mps'). Auto-detects if not specified.")
    return parser.parse_args()


def main():
    args = parse_args()

    device = get_device(args.device) if args.device else get_device()
    print(f"Using device: {device}")

    class_names = load_classes(args.classes)
    num_classes_actual = len(class_names)
    print(f"Loaded {num_classes_actual} classes: {class_names}")

    model = load_model_for_inference(args.model_path, args.model_type, num_classes_actual, device)

    model_type_lower = args.model_type.lower()  # Pre-calculate lower case model type
    image_processor = None
    if model_type_lower == "deformable_detr":
        if isinstance(model, DeformableDETRWrapper):
            image_processor = model.image_processor
        else:
            print("Warning: Deformable DETR model loaded but wrapper could not find image_processor.", file=sys.stderr)
            # Attempt fallback loading (requires transformers library)
            try:
                from transformers import AutoImageProcessor
                image_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr", use_fast=True)
                print("Warning: Loaded image processor independently as fallback.")
            except Exception as e:
                print(
                    f"Warning: Failed to load Deformable DETR image processor independently ({e}). Inference might fail.",
                    file=sys.stderr)

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    normalize_boxes_for_dataset = (model_type_lower == "deformable_detr")
    # Note: transform_fn is mainly for the non-DETR tensor creation in TestDataset
    transform_fn = get_simple_transform()

    try:
        test_dataset = TestDataset(
            images_dir=args.input, labels_dir=args.labels, classes_file=args.classes,
            transforms=transform_fn, normalize_boxes=normalize_boxes_for_dataset
        )
    except FileNotFoundError:
        print(f"Error: Input image directory '{args.input}' or label directory '{args.labels}' not found.",
              file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error creating TestDataset: {e}", file=sys.stderr)
        sys.exit(1)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0])
    print(f"Found {len(test_dataset)} images to process.")

    all_comparison_images = []
    max_width = 0

    with torch.no_grad():
        for i, (original_pil_image, image_tensor, gt_boxes_raw, gt_labels_raw) in enumerate(
                tqdm(test_loader, desc="Processing images")):
            try:
                base_filename = os.path.basename(test_dataset.image_files[i])
            except IndexError:
                base_filename = f"image_{i}"

            try:
                orig_w, orig_h = original_pil_image.size
                orig_size_list = [[orig_h, orig_w]]

                # Prepare model input & run inference
                if model_type_lower == "deformable_detr":
                    if image_processor:
                        # Use DETR's specific processor on the PIL image
                        hf_inputs = image_processor(images=original_pil_image, return_tensors="pt")
                        pixel_values = hf_inputs['pixel_values'].to(device)
                        predictions = model(pixel_values, orig_sizes=orig_size_list)
                    else:
                        print(
                            f"Error: Deformable DETR requires its image processor for {base_filename}, but it's unavailable. Skipping.",
                            file=sys.stderr)
                        continue  # Skip this image if processor missing
                else:
                    # Use the tensor generated by TestDataset's transform for other models
                    input_tensor_batch = image_tensor.unsqueeze(0).to(device)
                    predictions = model(input_tensor_batch)

                # Extract & Filter Predictions
                if not predictions:
                    pred_boxes, pred_labels, pred_scores = torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,))
                else:
                    pred = predictions[0]
                    pred_boxes = pred['boxes'].cpu()
                    pred_labels = pred['labels'].cpu()
                    pred_scores = pred['scores'].cpu()

                keep_indices = pred_scores >= args.threshold
                filtered_pred_boxes = pred_boxes[keep_indices]
                filtered_pred_labels = pred_labels[keep_indices]

                # Process Ground Truth Boxes/Labels
                if not isinstance(gt_boxes_raw, torch.Tensor): gt_boxes_raw = torch.tensor(gt_boxes_raw)
                if not isinstance(gt_labels_raw, torch.Tensor): gt_labels_raw = torch.tensor(gt_labels_raw)
                gt_labels_list = gt_labels_raw.cpu()

                gt_boxes_absolute = torch.empty((0, 4))  # Default empty
                if gt_boxes_raw.numel() > 0:
                    if normalize_boxes_for_dataset:  # Convert DETR's normalized GT boxes
                        boxes_cxcywh = gt_boxes_raw.cpu()
                        boxes_xyxy = torch.zeros_like(boxes_cxcywh)
                        boxes_xyxy[:, 0] = (boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2) * orig_w
                        boxes_xyxy[:, 1] = (boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2) * orig_h
                        boxes_xyxy[:, 2] = (boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2) * orig_w
                        boxes_xyxy[:, 3] = (boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2) * orig_h
                        gt_boxes_absolute = boxes_xyxy
                    else:  # Use absolute GT boxes directly
                        gt_boxes_absolute = gt_boxes_raw.cpu()

                # Draw GT and Prediction images
                gt_image = original_pil_image.copy()
                pred_image = original_pil_image.copy()
                pred_shift_labels = (model_type_lower != "deformable_detr")

                gt_image = evaluate_draw_boxes(gt_image, gt_boxes_absolute, gt_labels_list, class_names,
                                               shift_labels=False, color="lime")
                pred_image = evaluate_draw_boxes(pred_image, filtered_pred_boxes, filtered_pred_labels, class_names,
                                                 shift_labels=pred_shift_labels, color="red")

                # Combine side-by-side
                width, height = original_pil_image.size
                combined_image = Image.new("RGB", (width * 2, height))
                combined_image.paste(gt_image, (0, 0))
                combined_image.paste(pred_image, (width, 0))

                # Add titles
                draw_combined = ImageDraw.Draw(combined_image)
                try:
                    title_font = ImageFont.truetype("arial.ttf", size=20)
                except IOError:
                    title_font = ImageFont.load_default()
                title_y = min(10, height - 25) if height > 30 else 2
                draw_combined.text((10, title_y), "Ground Truth", fill="lime", font=title_font)
                draw_combined.text((width + 10, title_y), "Prediction", fill="red", font=title_font)

                all_comparison_images.append(combined_image)
                max_width = max(max_width, combined_image.width)

            except FileNotFoundError:
                print(f"Skipping - Input image file not found during processing: {base_filename}", file=sys.stderr)
            except Exception as e:
                print(f"Failed to process {base_filename}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()

    # Combine all images into one file after the loop
    if not all_comparison_images:
        print("No comparison images were generated.")
    else:
        print(f"\nCombining {len(all_comparison_images)} comparison images into a single file...")
        total_height = sum(img.height for img in all_comparison_images)
        final_composite_image = Image.new("RGB", (max_width, total_height), color="white")

        y_offset = 0
        for img in tqdm(all_comparison_images, desc="Stitching images"):
            final_composite_image.paste(img, (0, y_offset))
            y_offset += img.height

        model_basename = os.path.splitext(os.path.basename(args.model_path))[0]
        model_basename = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in model_basename)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        final_filename = f"all_comparisons_{model_basename}_{timestamp}.png"
        final_output_path = os.path.join(output_dir, final_filename)

        try:
            final_composite_image.save(final_output_path)
            print(f"\nSuccessfully saved combined comparison to: {final_output_path}")
        except Exception as e:
            print(f"\nError saving final composite image: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
