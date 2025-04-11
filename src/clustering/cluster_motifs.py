# src/analysis/cluster_motifs.py

import os
import sys
import argparse
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, UnidentifiedImageError
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
# For optional t-SNE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import shutil
import torch.nn as nn  # For nn.Identity

# --- START: Add project root to sys.path ---
# Allows imports like 'from src.models...' if needed, and consistent execution
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # src/analysis
_SRC_DIR = os.path.dirname(_SCRIPT_DIR)  # src
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)  # RockArtDetection
if _PROJECT_ROOT not in sys.path:
	sys.path.insert(0, _PROJECT_ROOT)
# --- END: Add project root to sys.path ---

# Potentially import helper functions if needed (e.g., get_device)
try:
	from src.training.utils import get_device
except ImportError:
	print("Warning: Could not import get_device from src.training.utils. Using manual device selection.",
		  file=sys.stderr)


	def get_device(device_arg=None):
		if device_arg:
			return torch.device(device_arg)
		if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
			return torch.device("mps")
		elif torch.cuda.is_available():
			return torch.device("cuda")
		else:
			return torch.device("cpu")


# --- Feature Extractor Loading ---
def load_feature_extractor(model_name, device):
	"""Loads a pre-trained torchvision model adapted for feature extraction."""
	model = None
	weights = 'DEFAULT'  # Use default weights (usually best available ImageNet pre-trained)

	# Standard ImageNet normalization
	preprocess = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	model_name_lower = model_name.lower()

	try:
		if model_name_lower == 'resnet18':
			model = models.resnet18(weights=weights)
			model.fc = nn.Identity()  # Remove final classification layer
		elif model_name_lower == 'resnet50':
			model = models.resnet50(weights=weights)
			model.fc = nn.Identity()
		elif model_name_lower == 'vgg16':
			model = models.vgg16(weights=weights)
			model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])  # Remove final layer
		elif model_name_lower == 'vgg19':
			model = models.vgg19(weights=weights)
			model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
		elif model_name_lower == 'densenet121':
			model = models.densenet121(weights=weights)
			model.classifier = nn.Identity()
		elif model_name_lower == 'inceptionv3':
			model = models.inception_v3(weights=weights, aux_logits=True)
			model.fc = nn.Identity()
			# InceptionV3 needs 299x299 input
			preprocess = transforms.Compose([
				transforms.Resize(299),
				transforms.CenterCrop(299),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			])
		else:
			raise ValueError(f"Unsupported feature extractor model: {model_name}")

		model.eval()
		model.to(device)
		print(f"Loaded feature extractor: {model_name} on {device}")
		return model, preprocess

	except Exception as e:
		print(f"Error loading model {model_name}: {e}", file=sys.stderr)
		sys.exit(1)


# --- Dataset for Cropped Images ---
class MotifDataset(Dataset):
	"""Simple dataset to load motif images from a directory."""

	def __init__(self, image_dir, transform=None):
		self.image_dir = image_dir
		self.transform = transform
		self.image_files = sorted([
			f for f in os.listdir(image_dir)
			if f.lower().endswith(('.png', '.jpg', '.jpeg'))
		])
		if not self.image_files:
			raise FileNotFoundError(f"No valid image files found in directory: {image_dir}")

	def __len__(self):
		return len(self.image_files)

	def __getitem__(self, idx):
		img_name = self.image_files[idx]
		img_path = os.path.join(self.image_dir, img_name)
		try:
			image = Image.open(img_path).convert('RGB')
			if self.transform:
				image = self.transform(image)
			return image, img_name
		except UnidentifiedImageError:
			print(f"Warning: Skipping corrupted or unreadable image: {img_path}", file=sys.stderr)
			# Return None or handle appropriately; here we return dummy data
			# A better approach might filter these out beforehand or handle None in the DataLoader collate_fn
			return torch.zeros((3, 224, 224)), f"CORRUPTED_{img_name}"  # Ensure size matches transforms
		except Exception as e:
			print(f"Error loading image {img_path}: {e}", file=sys.stderr)
			return torch.zeros((3, 224, 224)), f"ERROR_{img_name}"


# Filter out None items potentially returned by Dataset's __getitem__ on error
def collate_filter_none(batch):
	# Filter out None items first, or items with dummy tensors/error filenames
	batch = [item for item in batch if item is not None and not item[1].startswith(("CORRUPTED_", "ERROR_"))]
	if not batch:  # If the whole batch was bad
		return None, None  # Return None to signal skipping
	# If batch is valid, proceed with default collate logic (stack tensors, keep names as list)
	images = torch.stack([item[0] for item in batch], 0)
	filenames = [item[1] for item in batch]
	return images, filenames


# --- Feature Extraction Function ---
def extract_features(model, dataloader, device):
	"""Extracts features from images using the provided model."""
	features_list = []
	filenames_list = []
	model.eval()
	with torch.no_grad():
		for inputs, filenames in tqdm(dataloader, desc="Extracting features"):
			inputs = inputs.to(device)
			outputs = model(inputs)
			# If model output includes extra dimensions (e.g., batch), flatten per image
			# Example: Outputs might be (batch_size, feature_dim)
			features = outputs.cpu().numpy()
			features_list.append(features)
			filenames_list.extend(filenames)  # filenames is already a list/tuple from dataloader batch

	# Concatenate features from all batches
	all_features = np.concatenate(features_list, axis=0)
	# Filter out filenames corresponding to corrupted images if dummy data was returned
	valid_indices = [i for i, fname in enumerate(filenames_list) if not fname.startswith(("CORRUPTED_", "ERROR_"))]
	all_features = all_features[valid_indices]
	valid_filenames = [filenames_list[i] for i in valid_indices]

	return all_features, valid_filenames


# --- Clustering Function ---
def perform_clustering(features, algorithm_name, n_clusters, dbscan_eps=0.5, dbscan_min_samples=5):
	"""Performs clustering on the extracted features."""
	algorithm_name_lower = algorithm_name.lower()

	# Input validation for algorithms requiring n_clusters
	if algorithm_name_lower in ['kmeans', 'agglomerative', 'spectral'] and (n_clusters is None or n_clusters <= 0):
		raise ValueError(f"{algorithm_name} requires --num-clusters (K) > 0.")

	print(f"Performing {algorithm_name} clustering...")
	if algorithm_name_lower != 'dbscan':
		print(f"Target number of clusters (k): {n_clusters}")
	else:
		print(f"Using DBSCAN with eps={dbscan_eps}, min_samples={dbscan_min_samples}")

	if algorithm_name_lower == 'kmeans':
		# n_init='auto' is default in newer sklearn, suppresses warning
		try:
			model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
		except TypeError:  # Older sklearn might not have n_init='auto'
			model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
		labels = model.fit_predict(features)

	elif algorithm_name_lower == 'agglomerative':
		# Default linkage is 'ward', which requires Euclidean distance
		model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
		labels = model.fit_predict(features)

	elif algorithm_name_lower == 'spectral':
		# Can be computationally intensive for large datasets
		# Ensure n_clusters is less than n_samples
		if n_clusters >= features.shape[0]:
			print(
				f"Warning: n_clusters ({n_clusters}) >= n_samples ({features.shape[0]}). Adjusting n_clusters for Spectral Clustering.",
				file=sys.stderr)
			n_clusters = max(1, features.shape[0] - 1)  # Adjust k if needed

		model = SpectralClustering(n_clusters=n_clusters, random_state=42,
								   assign_labels='kmeans', affinity='nearest_neighbors',
								   n_neighbors=max(10, n_clusters))  # Ensure n_neighbors is reasonable
		labels = model.fit_predict(features)

	elif algorithm_name_lower == 'dbscan':
		# Does not use n_clusters argument, relies on eps and min_samples
		# Parameter tuning is CRUCIAL for DBSCAN, especially with high-dimensional features
		from sklearn.cluster import DBSCAN  # Import DBSCAN here
		model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric='euclidean')
		labels = model.fit_predict(features)
		# Note: DBSCAN labels noise points as -1.
		num_discovered_clusters = len(set(labels)) - (1 if -1 in labels else 0)
		num_noise_points = np.sum(labels == -1)
		print(f"DBSCAN discovered {num_discovered_clusters} clusters and {num_noise_points} noise points.")

	else:
		raise ValueError(f"Unsupported clustering algorithm: {algorithm_name}")

	print("Clustering complete.")
	return labels


# --- t-SNE Visualization Function (Optional) ---
def visualize_tsne(features, labels, filenames, output_path):
	"""Generates and saves a t-SNE plot of the features colored by cluster label."""
	print("Generating t-SNE visualization...")
	tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1), max_iter=300)
	features_tsne = tsne.fit_transform(features)

	plt.figure(figsize=(12, 10))
	scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='viridis', alpha=0.7)

	# Create a legend - handle case where labels might be -1 (noise for DBSCAN)
	unique_labels = np.unique(labels)
	if -1 in unique_labels:  # Handle noise points if using DBSCAN
		core_labels = [l for l in unique_labels if l != -1]
		handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {l}',
							  markerfacecolor=plt.cm.viridis(l / max(core_labels) if core_labels else 0), markersize=10)
				   for l in core_labels]
		handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Noise (-1)',
								  markerfacecolor='gray', markersize=10))
	else:
		handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {l}',
							  markerfacecolor=plt.cm.viridis(l / (max(unique_labels) if len(unique_labels) > 1 else 1)),
							  markersize=10) for l in unique_labels]

	plt.legend(handles=handles, title="Clusters")
	plt.title('t-SNE Visualization of Motif Clusters')
	plt.xlabel('t-SNE Dimension 1')
	plt.ylabel('t-SNE Dimension 2')
	plt.grid(True)

	try:
		plt.savefig(output_path, dpi=300)
		print(f"t-SNE plot saved to {output_path}")
	except Exception as e:
		print(f"Error saving t-SNE plot: {e}", file=sys.stderr)
	plt.close()  # Close plot to free memory


# --- Main Execution ---
def main():
	parser = argparse.ArgumentParser(description="Extract CNN features and cluster motif images.")
	parser.add_argument("--input", required=True, help="Directory containing cropped motif images.")
	parser.add_argument("--output", required=True,
						help="Directory to save clustering results (CSV, optional images, plot).")
	parser.add_argument("--feature-model", required=True,
						choices=['resnet18', 'resnet50', 'vgg16', 'vgg19', 'densenet121', 'inceptionv3'],
						help="CNN model for feature extraction.")
	parser.add_argument("--cluster-algo", required=True,
						choices=['kmeans', 'agglomerative', 'spectral', 'dbscan'],  # Added 'dbscan'
						help="Clustering algorithm to use.")
	parser.add_argument("--dbscan_eps", type=float, default=0.5,
						help="DBSCAN: The maximum distance between two samples for one to be considered as in the neighborhood of the other (adjust based on feature scale).")
	parser.add_argument("--dbscan_min_samples", type=int, default=5,
						help="DBSCAN: The number of samples in a neighborhood for a point to be considered as a core point.")
	parser.add_argument("--num-clusters", type=int, required=True, help="Number of clusters (K).")
	parser.add_argument("--batch-size", type=int, default=32, help="Batch size for feature extraction.")
	parser.add_argument("--device", default=None, help="Device ('cuda', 'cpu', 'mps'). Auto-detects if not specified.")
	parser.add_argument("--copy-images", action='store_true',
						help="Copy images into cluster-specific subdirectories in the output folder.")
	parser.add_argument("--visualize-tsne", action='store_true', help="Generate a t-SNE visualization plot.")

	args = parser.parse_args()

	# Setup
	if args.device:
		device = torch.device(args.device)
		print(f"Using specified device: {device}")
	else:
		# Call get_device() with no arguments if user didn't specify one
		device = get_device()  # From src.training.utils
		print(f"Using automatically detected device: {device}")
	os.makedirs(args.output, exist_ok=True)

	# Load Model and Preprocessor
	feature_extractor, preprocess = load_feature_extractor(args.feature_model, device)

	# Create Dataset and DataLoader
	try:
		dataset = MotifDataset(args.input, transform=preprocess)
	except FileNotFoundError as e:
		print(f"Error: {e}", file=sys.stderr)
		sys.exit(1)

	dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
							pin_memory=True if device.type == 'cuda' else False, collate_fn=collate_filter_none)

	# Extract Features
	features, filenames = extract_features(feature_extractor, dataloader, device)

	if len(features) == 0:
		print("Error: No features were extracted. Check input directory and image files.", file=sys.stderr)
		sys.exit(1)

	print(f"Extracted {features.shape[1]}-dimensional features for {len(filenames)} motifs.")

	# Perform Clustering
	cluster_labels = perform_clustering(features, args.cluster_algo, args.num_clusters)

	# Save Results
	results_df = pd.DataFrame({'filename': filenames, 'cluster_id': cluster_labels})
	csv_path = os.path.join(args.output,
							f'cluster_results_{args.feature_model}_{args.cluster_algo}_k{args.num_clusters}.csv')
	try:
		results_df.to_csv(csv_path, index=False)
		print(f"Cluster assignments saved to: {csv_path}")
	except Exception as e:
		print(f"Error saving results CSV: {e}", file=sys.stderr)

	# Optional: Copy images to cluster folders
	if args.copy_images:
		print("Copying images to cluster subdirectories...")
		cluster_dirs = {}
		for cluster_id in np.unique(cluster_labels):
			cluster_dir = os.path.join(args.output, f'cluster_{cluster_id}')
			os.makedirs(cluster_dir, exist_ok=True)
			cluster_dirs[cluster_id] = cluster_dir

		for filename, cluster_id in tqdm(zip(filenames, cluster_labels), total=len(filenames), desc="Copying images"):
			src_path = os.path.join(args.input, filename)
			dst_path = os.path.join(cluster_dirs[cluster_id], filename)
			try:
				if os.path.exists(src_path):
					shutil.copy2(src_path, dst_path)
				else:
					print(f"Warning: Source image not found during copy: {src_path}", file=sys.stderr)
			except Exception as e:
				print(f"Error copying {filename} to cluster {cluster_id}: {e}", file=sys.stderr)
		print("Image copying complete.")

	# Optional: Generate t-SNE plot
	if args.visualize_tsne:
		tsne_plot_path = os.path.join(args.output,
									  f'tsne_visualization_{args.feature_model}_{args.cluster_algo}_k{args.num_clusters}.png')
		visualize_tsne(features, cluster_labels, filenames, tsne_plot_path)

	print("\nClustering process finished.")


if __name__ == "__main__":
	main()
