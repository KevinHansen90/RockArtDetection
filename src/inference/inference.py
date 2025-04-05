# src/inference/inference.py

import argparse
import os
import sys
import time
import random
import torch
import torchvision.transforms as T
from torchvision.ops import nms # Non-Maximum Suppression if needed
from PIL import Image, ImageDraw, ImageFont
import cv2 # OpenCV for drawing
import numpy as np
from tqdm import tqdm

# Adjust import paths based on the new script location
try:
    # Assumes running from the project root (e.g., python src/inference/inference.py)
    from src.models.detection_models import get_detection_model, DeformableDETRWrapper
    from src.datasets.yolo_dataset import load_classes
    from src.training.utils import get_device, get_simple_transform
except ModuleNotFoundError:
    # Fallback if the above fails (e.g., different execution context)
    # May need further adjustment depending on how the script is invoked
    print("Warning: Using fallback imports. Ensure PYTHONPATH is set correctly or run from project root.", file=sys.stderr)
    # Attempt imports relative to the potential src directory if added to path
    from models.detection_models import get_detection_model, DeformableDETRWrapper
    from datasets.yolo_dataset import load_classes
    from training.utils import get_device, get_simple_transform


# --- (Rest of the script remains the same as provided previously) ---

# Colors for bounding boxes
BOX_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0),
    (128, 128, 0), (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128)
]

def load_model_for_inference(model_path, model_type, num_classes_actual, device):
    """Loads model architecture, state dict, sets to eval mode."""
    model_type = model_type.lower()
    # Determine num_classes needed for model initialization
    if model_type == "deformable_detr":
        num_classes_model = num_classes_actual # DETR expects actual class count
    else:
        num_classes_model = num_classes_actual + 1 # Torchvision models expect background class

    # Load model structure
    # Pass dummy config, specific config params not usually needed for inference structure
    model = get_detection_model(model_type, num_classes_model, config={})

    # Load the checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
        # Handle potential keys like 'model_state_dict' or just the state_dict
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
    model.eval() # Set model to evaluation mode
    print(f"Model '{model_type}' loaded on device '{device}' and set to evaluation mode.")
    return model

def preprocess_image(image_path, device, model_type, image_processor=None):
    """Loads and preprocesses a single image."""
    pil_image = Image.open(image_path).convert("RGB")
    model_type = model_type.lower()

    if model_type == "deformable_detr":
        if image_processor is None:
             raise ValueError("Image processor must be provided for Deformable DETR.")
        # Deformable DETR uses its own processor
        inputs = image_processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        return pixel_values, pil_image
    else:
        # Use standard torchvision transform
        transform = get_simple_transform()
        image_tensor = transform(pil_image).to(device)
        return image_tensor.unsqueeze(0), pil_image # Add batch dimension


def draw_predictions(pil_image, boxes, labels, scores, class_names, threshold, model_type):
    """Draws bounding boxes and labels on the image."""
    # Convert PIL image to OpenCV format (RGB -> BGR)
    open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    height, width, _ = open_cv_image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    model_type = model_type.lower()
    # Adjust class index if model output includes background class (index 0)
    label_offset = 0 if model_type == "deformable_detr" else -1

    indices_to_keep = scores >= threshold
    filtered_boxes = boxes[indices_to_keep]
    filtered_labels = labels[indices_to_keep]
    filtered_scores = scores[indices_to_keep]

    print(f"  Found {len(filtered_boxes)} detections above threshold {threshold}.")

    for i, (box, label_idx, score) in enumerate(zip(filtered_boxes, filtered_labels, filtered_scores)):
        # Denormalize if boxes are normalized (e.g., DETR often returns normalized cxcywh)
        # Assuming boxes are [x1, y1, x2, y2] absolute for torchvision,
        # and potentially normalized [x1, y1, x2, y2] for DETR post-processing
        if model_type == "deformable_detr":
             # Assuming DETR output after post_process_object_detection is [x_min, y_min, x_max, y_max] absolute
             # Verify this assumption based on the specific HF processor version if results look wrong.
            pass # Assuming absolute coordinates already

        x1, y1, x2, y2 = map(int, box)

        # Clamp box coordinates to image dimensions
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        # Get class name and color
        class_idx = label_idx.item() + label_offset
        if 0 <= class_idx < len(class_names):
            class_name = class_names[class_idx]
            color = BOX_COLORS[class_idx % len(BOX_COLORS)]
        else:
            class_name = f"UNKNOWN_IDX_{label_idx.item()}"
            color = (100, 100, 100) # Grey for unknown

        # Draw bounding box
        cv2.rectangle(open_cv_image, (x1, y1), (x2, y2), color, thickness)

        # Prepare label text
        label_text = f"{class_name}: {score:.2f}"

        # Get text size to draw background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        # Draw background rectangle for text
        # Position above the box, but clamp within image bounds
        label_y = max(text_height + baseline, y1 - baseline)
        label_x = x1
        cv2.rectangle(open_cv_image, (label_x, label_y - text_height - baseline),
                      (label_x + text_width, label_y), color, -1) # Filled rectangle

        # Draw text
        cv2.putText(open_cv_image, label_text, (label_x, label_y - baseline // 2),
                    font, font_scale, (255, 255, 255), 1, cv2.LINE_AA) # White text

    # Convert back to PIL format (BGR -> RGB) for saving
    return Image.fromarray(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB))


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on images using a trained detection model.")
    parser.add_argument("--model-path", required=True, help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("--input", required=True, help="Path to a single input image or a directory of images.")
    parser.add_argument("--output", required=True, help="Directory to save output images with detections.")
    parser.add_argument("--classes", required=True, help="Path to the file containing class names (one per line).")
    parser.add_argument("--model-type", required=True, choices=["fasterrcnn", "retinanet", "deformable_detr"], help="Type of model architecture used.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for displaying detections.")
    parser.add_argument("--device", default=None, help="Device to use ('cuda', 'cpu', 'mps'). Auto-detects if not specified.")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Setup Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"Using device: {device}")

    # 2. Load Class Names
    class_names = load_classes(args.classes)
    num_classes_actual = len(class_names)
    print(f"Loaded {num_classes_actual} classes: {class_names}")

    # 3. Load Model
    model = load_model_for_inference(args.model_path, args.model_type, num_classes_actual, device)
    image_processor = None
    if args.model_type == "deformable_detr" and isinstance(model, DeformableDETRWrapper):
         image_processor = model.image_processor # Get processor from DETR wrapper

    # 4. Prepare Output Directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Use basename of model file (without extension) for potentially clearer output folder name
    model_basename = os.path.splitext(os.path.basename(args.model_path))[0]
    output_subdir = os.path.join(args.output, f"inference_{model_basename}_{timestamp}")
    os.makedirs(output_subdir, exist_ok=True)
    print(f"Output will be saved to: {output_subdir}")

    # 5. Find Input Images
    image_paths = []
    if os.path.isdir(args.input):
        for fname in os.listdir(args.input):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                image_paths.append(os.path.join(args.input, fname))
    elif os.path.isfile(args.input):
        image_paths.append(args.input)
    else:
        raise FileNotFoundError(f"Input path {args.input} is not a valid file or directory.")

    if not image_paths:
        print("No images found in the input path.")
        return

    print(f"Found {len(image_paths)} images to process.")

    # 6. Process Images
    with torch.no_grad(): # Disable gradient calculations for inference
        for img_path in tqdm(image_paths, desc="Processing images"):
            base_filename = os.path.basename(img_path)
            output_filename = os.path.splitext(base_filename)[0] + "_pred.jpg"
            output_path = os.path.join(output_subdir, output_filename)

            try:
                # Preprocess
                input_tensor, original_pil_image = preprocess_image(img_path, device, args.model_type, image_processor)

                # Inference
                start_time = time.time()
                predictions = model(input_tensor)
                end_time = time.time()
                inf_time = end_time - start_time
                print(f"  Inference time for {base_filename}: {inf_time:.4f} seconds")

                # Predictions format varies:
                # Torchvision: List of dicts, one per image. Each dict has 'boxes', 'labels', 'scores'.
                # DeformableDETRWrapper: List of dicts (already post-processed by HF processor)

                # Assuming batch size 1 for inference here
                pred = predictions[0]

                # Extract results (ensure they are on CPU for drawing)
                boxes = pred['boxes'].cpu()
                labels = pred['labels'].cpu()
                scores = pred['scores'].cpu()

                # Draw predictions
                result_image = draw_predictions(original_pil_image, boxes, labels, scores, class_names, args.threshold, args.model_type)

                # Save result
                result_image.save(output_path)
                # print(f"  Saved prediction to {output_path}")

            except Exception as e:
                print(f"  Failed to process {base_filename}: {e}")

    print(f"Inference complete. Results saved in {output_subdir}")


if __name__ == "__main__":
    main()