#!/usr/bin/env python3
from __future__ import annotations

import logging
from typing import List, Sequence

import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
_FONT = ImageFont.load_default()


def _load_font(path: str = "arial.ttf", size: int = 15) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size=size)
    except IOError:
        log.warning("Font '%s' not found; using default.", path)
        return _FONT


FONT = _load_font()


def _label_to_class_idx(label: int, model_type: str) -> int:
    """FR-CNN / RetinaNet reserve 0 as background → subtract 1 for class lookup."""
    return label - 1 if model_type in {"fasterrcnn", "retinanet"} else label


def _cxcywh_to_xyxy(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
    """Convert normalised cxcywh → absolute xyxy."""
    cx, cy, bw, bh = boxes.T
    x0 = (cx - bw / 2) * w
    y0 = (cy - bh / 2) * h
    x1 = (cx + bw / 2) * w
    y1 = (cy + bh / 2) * h
    return torch.stack([x0, y0, x1, y1], dim=1)


# --------------------------------------------------------------------------- #
# Drawing helper                                                              #
# --------------------------------------------------------------------------- #
def _draw_boxes(
    image: Image.Image,
    boxes: Sequence[Sequence[float]],
    labels: Sequence[int],
    classes: List[str],
    *,
    color: str,
    margin: int = 3,
    font: ImageFont.FreeTypeFont = FONT,
) -> Image.Image:
    draw = ImageDraw.Draw(image)

    for box, lab in zip(boxes, labels):
        try:
            x0, y0, x1, y1 = [int(round(float(c))) for c in box]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = max(x0, x1), max(y0, y1)
        except Exception as e:
            log.warning("Invalid box %s (%s) – skipped", box, e)
            continue

        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

        cls_txt = classes[lab] if 0 <= lab < len(classes) else f"UNK={lab}"
        try:
            tw, th = draw.textbbox((0, 0), cls_txt, font=font)[2:]
        except Exception:
            tw, th = font.getsize(cls_txt)

        rx0, ry0 = x0, max(0, y0 - th - 2 * margin)
        draw.rectangle([rx0, ry0, rx0 + tw + 2 * margin, y0], fill=color)
        draw.text((rx0 + margin, ry0 + margin // 2), cls_txt, fill="white", font=font)

    return image


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def evaluate_and_visualize(
    model,
    test_loader,
    classes: List[str],
    device: torch.device,
    output_path: str,
    *,
    threshold: float = 0.5,
    model_type: str = "fasterrcnn",
    max_images: int = 10,
) -> None:
    """
    Run inference on *test_loader* and save a side-by-side GT vs prediction
    collage to *output_path*.
    """
    mt = model_type.lower()
    is_frcnn = mt in {"fasterrcnn", "retinanet"}
    is_detr = mt == "deformable_detr"

    model.eval().to(device)
    visualizations: List[Image.Image] = []

    for idx, (pil_img, img_tensor, gt_boxes, gt_labels) in enumerate(
        tqdm(test_loader, desc="Eval-vis", leave=False)
    ):
        if idx >= max_images:
            break

        # ---------- forward pass -------------------------------------- #
        with torch.inference_mode():
            pred = model(img_tensor.unsqueeze(0).to(device))[0]

        # ---------- filter predictions -------------------------------- #
        keep = pred["scores"].cpu() >= threshold
        if is_frcnn:                  # drop background label 0
            keep &= pred["labels"].cpu() != 0
        pred_boxes = pred["boxes"].cpu()[keep].tolist()
        pred_labels = [
            _label_to_class_idx(int(lab), mt) for lab in pred["labels"].cpu()[keep]
        ]

        # ---------- ground-truth processing --------------------------- #
        gt_boxes_t = gt_boxes if isinstance(gt_boxes, torch.Tensor) else torch.tensor(gt_boxes)
        if is_detr and gt_boxes_t.numel():
            w, h = pil_img.size
            gt_boxes_t = _cxcywh_to_xyxy(gt_boxes_t, w, h)

        gt_boxes_list = gt_boxes_t.tolist()
        gt_labels_list = [_label_to_class_idx(int(l), mt) for l in (
            gt_labels.tolist() if isinstance(gt_labels, torch.Tensor) else gt_labels
        )]

        # ---------- draw ---------------------------------------------- #
        gt_vis   = _draw_boxes(pil_img.copy(), gt_boxes_list, gt_labels_list, classes, color="green")
        pred_vis = _draw_boxes(pil_img.copy(), pred_boxes,    pred_labels,    classes, color="red")

        w, h = pil_img.size
        combined = Image.new("RGB", (w * 2, h))
        combined.paste(gt_vis, (0, 0))
        combined.paste(pred_vis, (w, 0))
        visualizations.append(combined)

    if not visualizations:
        log.info("No images processed – nothing to save.")
        return

    out_h = sum(img.height for img in visualizations)
    out_w = max(img.width for img in visualizations)
    collage = Image.new("RGB", (out_w, out_h))

    y = 0
    for img in visualizations:
        collage.paste(img, (0, y))
        y += img.height

    collage.save(output_path)
    log.info("Saved visualization → %s", output_path)
