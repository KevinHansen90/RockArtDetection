import os
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
import numpy as np


# Define the function to load and preprocess the image
def load_image(image_path):
	return Image.open(image_path)


# Function to save extracted features
def save_features(features, output_path):
	np.save(output_path, features.cpu().numpy())


# Load the pre-trained ViT model and processor
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
model.eval()

# Directories
image_dir = '../data/raw'
output_dir = '../output/features'

# Ensure output directory exists
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

# Run feature extraction on the dataset
for idx, filename in enumerate(os.listdir(image_dir)):
	if filename.endswith((".jpg", ".jpeg", ".png", ".JPG")):
		image_path = os.path.join(image_dir, filename)
		image = load_image(image_path)

		# Preprocess the image
		inputs = processor(images=image, return_tensors="pt")

		# Extract features
		with torch.no_grad():
			outputs = model(**inputs)

		# Get the last hidden states
		last_hidden_states = outputs.last_hidden_state

		# Save the extracted features
		output_path = os.path.join(output_dir, f"features_{idx + 1}.npy")
		save_features(last_hidden_states, output_path)
		print(f"Saved features to {output_path}")

print("Feature extraction complete.")
