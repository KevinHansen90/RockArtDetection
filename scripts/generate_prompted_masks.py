import os
import torch
from PIL import Image
from transformers import SamModel, SamProcessor
import numpy as np
import matplotlib.pyplot as plt


# Define the function to load and preprocess the image
def load_image(image_path):
	return Image.open(image_path)


# Function to load annotations
def load_annotations(annotation_path):
	with open(annotation_path, 'r') as f:
		annotations = [list(map(float, line.strip().split())) for line in f]
	return annotations


# Function to visualize and save the mask
def visualize_and_save_mask(image, mask, output_path):
	# Convert the mask to a 2D array if it's not already
	if mask.ndim == 3:
		mask = mask[0]

	# Visualize the mask
	plt.figure(figsize=(10, 10))
	plt.imshow(image)
	plt.imshow(mask, alpha=0.5, cmap='jet')
	plt.axis('off')
	plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
	plt.close()


# Load the pre-trained SAM model and processor
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
model = SamModel.from_pretrained("facebook/sam-vit-huge")

# Directories
image_dir = '../data/raw'
annotation_dir = '../data/annotations'
output_dir = '../output/masks'

# Ensure output directory exists
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

# Run mask generation on the dataset
for idx, filename in enumerate(os.listdir(image_dir)):
	if filename.endswith((".jpg", ".jpeg", ".png", ".JPG")):
		image_path = os.path.join(image_dir, filename)
		annotation_path = os.path.join(annotation_dir, filename.rsplit('.', 1)[0] + '.txt')

		image = load_image(image_path).convert("RGB")
		annotations = load_annotations(annotation_path)

		mask = np.zeros((image.height, image.width), dtype=np.uint8)

		input_points = []
		for annotation in annotations:
			# Extract the bounding box center points
			_, x_center, y_center, width, height = annotation
			x_center = int(x_center * image.width)
			y_center = int(y_center * image.height)
			input_points.append([x_center, y_center])

		if input_points:
			inputs = processor(images=image, input_points=[input_points], return_tensors="pt")

			# Run inference
			with torch.no_grad():
				outputs = model(**inputs)

			# Get the predicted mask
			predicted_masks = processor.image_processor.post_process_masks(
				outputs.pred_masks,
				inputs["original_sizes"],
				inputs["reshaped_input_sizes"]
			)[0]

			# Combine masks
			for predicted_mask in predicted_masks:
				mask = np.maximum(mask, predicted_mask.numpy())

		# Save the mask as an image
		mask_path = os.path.join(output_dir, f"mask_{idx + 1}.png")
		visualize_and_save_mask(image, mask, mask_path)
		print(f"Saved mask to {mask_path}")

print("Mask generation complete.")
