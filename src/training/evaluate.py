# src/training/evaluate.py (Modified function)

import torch
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont


def draw_boxes(image, boxes, labels, classes, shift_labels=True, color="red"):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=15)
    except Exception:
        font = ImageFont.load_default()

    for box, label in zip(boxes, labels):
        # Convert tensor to list if needed.
        if isinstance(box, torch.Tensor):
            box = box.tolist()
        # Ensure box coordinates are integers for drawing
        box_int = [int(round(coord)) for coord in box]
        draw.rectangle(box_int, outline=color, width=2)

        # Decide how to compute the class index.
        # For models that shifted labels (FasterRCNN/RetinaNet), subtract 1.
        # For deformable_detr, use label as-is.
        class_index = int((label - 1) if shift_labels else label) # Ensure label is int

        if 0 <= class_index < len(classes):
            text = classes[class_index]
        else:
            text = f"ID={label}" # Use original label if index out of bounds

        # Get text size to potentially add background rectangle
        try:
            # text_bbox = font.getbbox(text) # Requires Pillow >= 9.5.0
            # text_width = text_bbox[2] - text_bbox[0]
            # text_height = text_bbox[3] - text_bbox[1]
            text_width, text_height = font.getsize(text) # Legacy method
        except AttributeError: # Fallback for older Pillow or default font
             text_width, text_height = font.getsize(text)

        text_location = (box_int[0], box_int[1] - text_height - 1) # Place above box
        # Optional: add background rectangle for better visibility
        # draw.rectangle((text_location[0], text_location[1], text_location[0] + text_width, text_location[1] + text_height), fill=color)
        draw.text(text_location, text, fill=color, font=font) # Changed fill to color for consistency, consider black/white
    return image


def evaluate_and_visualize(model, test_loader, classes, device, output_path, threshold=0.5, model_type="fasterrcnn"):
    model.eval()
    model.to(device)

    # Determine if labels need shifting (True for FasterRCNN/RetinaNet, False for DETR)
    shift_labels = (model_type != "deformable_detr")

    visualizations = []
    all_pred_scores = []
    all_pred_labels = []

    for (pil_img, img_tensor, gt_boxes_raw, gt_labels_raw) in tqdm(test_loader, desc="Evaluating on test set", leave=False):
        img_tensor = img_tensor.unsqueeze(0).to(device)
        orig_w, orig_h = pil_img.size # Get original image dimensions

        with torch.no_grad():
            preds = model(img_tensor)[0] # Get predictions for the single image

        pred_boxes  = preds["boxes"]  # shape [N, 4], expected absolute [x1,y1,x2,y2]
        pred_scores = preds["scores"] # shape [N]
        pred_labels = preds["labels"] # shape [N], expected 1-based for F-RCNN/Retina, 0-based for DETR

        # Filter predictions by threshold.
        keep_boxes, keep_labels, keep_scores = [], [], []
        for b, lbl, s in zip(pred_boxes, pred_labels, pred_scores):
            if s.item() >= threshold:
                keep_boxes.append(b) # Already absolute coords from model/wrapper output
                keep_labels.append(lbl.item())
                keep_scores.append(s.item())

        # --- Ground Truth Box Conversion (Specific for DETR) ---
        gt_boxes_absolute = []
        if model_type == "deformable_detr":
            # Convert DETR's normalized [cx, cy, w, h] GT boxes to absolute [x1, y1, x2, y2]
            for box_norm in gt_boxes_raw: # gt_boxes_raw is tensor [M, 4]
                 cx, cy, w, h = box_norm.tolist()
                 x1 = (cx - w / 2) * orig_w
                 y1 = (cy - h / 2) * orig_h
                 x2 = (cx + w / 2) * orig_w
                 y2 = (cy + h / 2) * orig_h
                 gt_boxes_absolute.append([x1, y1, x2, y2])
        else:
            # For other models, GT boxes are already absolute [x1, y1, x2, y2]
             gt_boxes_absolute = [b.tolist() if isinstance(b, torch.Tensor) else b for b in gt_boxes_raw]

        # Convert GT labels tensor to list of ints
        gt_labels_list = [lbl.item() for lbl in gt_labels_raw]

        # Draw ground truth in green, predictions in red.
        gt_image   = pil_img.copy()
        pred_image = pil_img.copy()

        # Use the processed absolute GT boxes for drawing
        gt_image = draw_boxes(gt_image, gt_boxes_absolute, gt_labels_list, classes, shift_labels=shift_labels, color="green")
        # Use the filtered predicted boxes (already absolute) for drawing
        pred_image = draw_boxes(pred_image, keep_boxes, keep_labels, classes, shift_labels=shift_labels, color="red")

        # Combine images side-by-side
        width, height = pil_img.size
        combined = Image.new("RGB", (width * 2, height))
        combined.paste(gt_image, (0, 0))
        combined.paste(pred_image, (width, 0))
        visualizations.append(combined)

        # Store scores and labels for potential further analysis (optional)
        all_pred_scores.extend(keep_scores)
        all_pred_labels.extend(keep_labels)

    # Combine all visualizations vertically and save the final image.
    if visualizations:
        total_height = sum(img.height for img in visualizations)
        max_width = max(img.width for img in visualizations) # Should be width*2
        final_img = Image.new("RGB", (max_width, total_height))
        y_offset = 0
        for vis in visualizations:
            final_img.paste(vis, (0, y_offset))
            y_offset += vis.height
        final_img.save(output_path)
        tqdm.write(f"Saved test visualization to {output_path}")
    else:
        tqdm.write("No test images processed or no predictions above threshold; skipping visualization saving.")

