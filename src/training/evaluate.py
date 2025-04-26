#!/usr/bin/env python3

import torch
import logging
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import sys

# Module logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Default drawing parameters
DEFAULT_FONT_PATH = "arial.ttf"
DEFAULT_FONT_SIZE = 15
DEFAULT_COLOR_GT = "green"
DEFAULT_COLOR_PRED = "red"
DEFAULT_MARGIN = 3


def _load_font(path=DEFAULT_FONT_PATH, size=DEFAULT_FONT_SIZE):
    try:
        return ImageFont.truetype(path, size=size)
    except IOError:
        logger.warning(f"Font '{path}' not found; using default font.")
        return ImageFont.load_default()


# Load font once
FONT = _load_font()


def draw_boxes(
    image: Image.Image,
    boxes,
    labels,
    classes: list,
    shift_labels: bool = True,
    color: str = DEFAULT_COLOR_PRED,
    margin: int = DEFAULT_MARGIN,
    font: ImageFont.FreeTypeFont = FONT
) -> Image.Image:
    """
    Draw bounding boxes with labels on an image.
    boxes: list of [x1, y1, x2, y2] or Tensor
    labels: list of int or Tensor
    classes: list of class names
    shift_labels: if True, subtract 1 from label index before lookup
    """
    draw = ImageDraw.Draw(image)

    for idx, (box, label) in enumerate(zip(boxes, labels)):
        # Convert box coords to ints, clamp and sort so x0<=x1, y0<=y1
        try:
            raw = box.tolist() if isinstance(box, torch.Tensor) else box
            coords = [int(round(float(c))) for c in raw]
            x0, y0, x1, y1 = coords
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = max(x0, x1)
            y1 = max(y0, y1)
        except Exception as e:
            logger.warning(f"Skipping invalid box at index {idx}: {box}. Error: {e}")
            continue

        # Draw the bounding box rectangle
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

        # Determine text label
        try:
            lab = label.item() if isinstance(label, torch.Tensor) else int(label)
            class_idx = (lab - 1) if shift_labels else lab
            text = classes[class_idx] if 0 <= class_idx < len(classes) else f"UNK_ID={lab}"
        except Exception as e:
            logger.warning(f"Skipping invalid label at index {idx}: {label}. Error: {e}")
            continue

        # Measure text size
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = font.getsize(text)

        # Compute and clamp background rectangle for text
        rx0 = x0
        ry0 = max(0, y0 - text_h - 2 * margin)
        rx1 = x0 + text_w + 2 * margin
        ry1 = y0
        rx0b, rx1b = sorted((rx0, rx1))
        ry0b, ry1b = sorted((ry0, ry1))

        # Draw background rectangle
        draw.rectangle([(rx0b, ry0b), (rx1b, ry1b)], fill=color)

        # Draw text
        text_x = rx0b + margin
        text_y = ry0b + margin // 2
        draw.text((text_x, text_y), text, fill="white", font=font)

    return image


def evaluate_and_visualize(
    model,
    test_loader,
    classes: list,
    device,
    output_path: str,
    threshold: float = 0.5,
    model_type: str = "fasterrcnn"
):
    """
    Run inference on test set and save side-by-side GT vs prediction visualizations.
    """
    model.eval()
    model.to(device)

    mt = model_type.lower()
    shift_labels = (mt == "fasterrcnn" or mt == "retinanet")

    visualizations = []
    max_vis = min(len(test_loader), 10)

    for i, batch in enumerate(tqdm(test_loader, desc="Eval Test", leave=False)):
        if i >= max_vis:
            break
        pil_img, img_tensor, gt_boxes_raw, gt_labels_raw = batch

        # Inference
        img = img_tensor.unsqueeze(0).to(device)
        with torch.inference_mode():
            preds = model(img)
            preds = preds[0] if isinstance(preds, list) else preds

        # --- Extract & filter predictions (apply threshold to every model) ---
        boxes = preds.get("boxes", torch.empty((0, 4))).cpu()
        scores = preds.get("scores", torch.empty((0,))).cpu()
        labels = preds.get("labels", torch.empty((0,))).cpu()
        # Filter by score AND filter out background class (label 0 for FR/Retina)
        score_keep = scores >= threshold
        if mt == "fasterrcnn" or mt == "retinanet":
            # For these models, label 0 is background, filter it out
            class_keep = labels != 0
            keep = score_keep & class_keep  # Combine score and non-background filters
        else:
            # For DETR, assume background is handled elsewhere or not applicable in the same way
            keep = score_keep
        pred_boxes = boxes[keep].tolist()
        pred_labels = labels[keep].tolist()

        # --- Process GT boxes & labels for DETR vs others ---
        # gt_boxes_raw is either absolute xyxy (for FR/Retina) or normalized cxcywh (for DETR)
        if isinstance(gt_boxes_raw, torch.Tensor):
            gt_boxes_tensor = gt_boxes_raw.cpu()
        else:
            gt_boxes_tensor = torch.tensor(gt_boxes_raw)

        if mt == "deformable_detr" and gt_boxes_tensor.numel():
            # Convert normalized cxcywh -> absolute xyxy
            img_w, img_h = pil_img.size  # PIL: (width, height)
            cx, cy, ww, hh = gt_boxes_tensor.T
            x0 = (cx - ww / 2) * img_w
            y0 = (cy - hh / 2) * img_h
            x1 = (cx + ww / 2) * img_w
            y1 = (cy + hh / 2) * img_h
            abs_boxes = torch.stack([x0, y0, x1, y1], dim=1)
            gt_boxes = abs_boxes.tolist()
        else:
            # Already in absolute xyxy for FR/Retina
            gt_boxes = gt_boxes_tensor.tolist()

        # GT labels come out shifted (+1) for FR/Retina, unshifted for DETR
        if isinstance(gt_labels_raw, torch.Tensor):
            gt_labels = gt_labels_raw.cpu().tolist()
        else:
            gt_labels = gt_labels_raw

        # --- Draw GT (green) & preds (red) with correct shift_labels logic ---
        gt_vis = draw_boxes(pil_img.copy(), gt_boxes, gt_labels,
                            classes, shift_labels, color="green")
        pred_vis = draw_boxes(pil_img.copy(), pred_boxes, pred_labels,
                              classes, shift_labels, color="red")

        combined = Image.new("RGB", (pil_img.width*2, pil_img.height))
        combined.paste(gt_vis,   (0, 0))
        combined.paste(pred_vis, (pil_img.width, 0))
        visualizations.append(combined)

    if not visualizations:
        logger.info("No visualizations created; skipping save.")
        return

    # Merge all visualizations vertically
    total_h = sum(img.height for img in visualizations)
    max_w   = max(img.width  for img in visualizations)
    final   = Image.new("RGB", (max_w, total_h))
    y_off   = 0
    for img in visualizations:
        final.paste(img, (0, y_off))
        y_off += img.height

    final.save(output_path)
    logger.info(f"Saved visualization: {output_path}")
