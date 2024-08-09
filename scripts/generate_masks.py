import os
import torch
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import numpy as np
import matplotlib.pyplot as plt

# Define the function to load and preprocess the image
def load_image(image_path):
    return Image.open(image_path)

# Function to visualize and save the mask
def visualize_and_save_mask(image, mask, output_path):
    # Visualize the mask
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5, cmap='jet')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Load the pre-trained segmentation model and processor
processor = SegformerImageProcessor.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')

# Directories
image_dir = '../data/processed'
output_dir = '../output/masks'

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Run mask generation on the dataset
for idx, filename in enumerate(os.listdir(image_dir)):
    if filename.endswith((".jpg", ".jpeg", ".png", ".JPG")):
        image_path = os.path.join(image_dir, filename)
        image = load_image(image_path)

        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt")

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the predicted mask
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(logits, size=image.size[::-1], mode='bilinear', align_corners=False)
        predicted_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        # Save the mask as an image
        mask_path = os.path.join(output_dir, f"mask_{idx + 1}.png")
        visualize_and_save_mask(image, predicted_mask, mask_path)
        print(f"Saved mask to {mask_path}")

print("Mask generation complete.")
