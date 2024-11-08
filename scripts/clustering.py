import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import math

# Directories containing the images and labels
images_dir = '/Users/kevinhansen/Documents/Git/RockArtDetection/data/train/images/'
labels_dir = '/Users/kevinhansen/Documents/Git/RockArtDetection/data/train/labels/'

# Feature extraction models to use
models_to_use = {
	'resnet18': models.resnet18(pretrained=True),
	'vgg16': models.vgg16(pretrained=True),
	'densenet121': models.densenet121(pretrained=True),
	'inception_v3': models.inception_v3(pretrained=True),
}

# Remove last layers and set to evaluation mode
for model_name, model in models_to_use.items():
	model.eval()
	if model_name == 'inception_v3':
		# Remove aux_logits
		model.aux_logits = False
		# Extract the feature extraction part
		models_to_use[model_name] = torch.nn.Sequential(
			model.Conv2d_1a_3x3,
			model.Conv2d_2a_3x3,
			model.Conv2d_2b_3x3,
			torch.nn.MaxPool2d(3, stride=2),
			model.Conv2d_3b_1x1,
			model.Conv2d_4a_3x3,
			torch.nn.MaxPool2d(3, stride=2),
			model.Mixed_5b,
			model.Mixed_5c,
			model.Mixed_5d,
			model.Mixed_6a,
			model.Mixed_6b,
			model.Mixed_6c,
			model.Mixed_6d,
			model.Mixed_6e,
			model.Mixed_7a,
			model.Mixed_7b,
			model.Mixed_7c,
			torch.nn.AdaptiveAvgPool2d((1, 1)),
			torch.nn.Flatten(),
		)
	else:
		# Remove the last classification layer
		models_to_use[model_name] = torch.nn.Sequential(
			*(list(model.children())[:-1]),
			torch.nn.Flatten()
		)

# Standard ImageNet normalization mean and std
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Preprocessing transforms per model
preprocess_transforms = {
	'resnet18': transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
	]),
	'vgg16': transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
	]),
	'densenet121': transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
	]),
	'inception_v3': transforms.Compose([
		transforms.Resize(342),
		transforms.CenterCrop(299),
		transforms.ToTensor(),
		transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
	]),
}

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for model_name, model in models_to_use.items():
	models_to_use[model_name] = model.to(device)

# List to store image data
image_data = []  # Each element: (img_name, idx, cropped_image)

# Iterate over all label files
for label_filename in os.listdir(labels_dir):
	if label_filename.endswith('.txt'):
		label_filepath = os.path.join(labels_dir, label_filename)
		image_filename = os.path.splitext(label_filename)[0] + '.jpg'  # Change extension if needed
		image_filepath = os.path.join(images_dir, image_filename)

		# Open the image
		if not os.path.exists(image_filepath):
			continue  # Skip if image does not exist
		image = Image.open(image_filepath).convert('RGB')
		image_width, image_height = image.size

		with open(label_filepath, 'r') as f:
			lines = f.readlines()
			for idx, line in enumerate(lines):
				tokens = line.strip().split()
				if len(tokens) == 5:
					class_id = int(tokens[0])
					if class_id == 1:  # Only process class '1' (animals)
						x_center = float(tokens[1])
						y_center = float(tokens[2])
						width = float(tokens[3])
						height = float(tokens[4])

						# Convert normalized coordinates to pixel values
						x_min = int((x_center - width / 2) * image_width)
						y_min = int((y_center - height / 2) * image_height)
						x_max = int((x_center + width / 2) * image_width)
						y_max = int((y_center + height / 2) * image_height)

						# Ensure coordinates are within image bounds
						x_min = max(0, x_min)
						y_min = max(0, y_min)
						x_max = min(image_width, x_max)
						y_max = min(image_height, y_max)

						# Crop the image to the bounding box
						cropped_image = image.crop((x_min, y_min, x_max, y_max))
						# Store image data with an identifier
						image_data.append((image_filename, idx, cropped_image))

# Clustering algorithms to use
clustering_algorithms = {
	'kmeans': KMeans,
	'agglomerative': AgglomerativeClustering,
	'dbscan': DBSCAN,
	'spectral': SpectralClustering
}

# Now, for each model, perform feature extraction and clustering
for model_name, model in models_to_use.items():
	print(f"\nProcessing with model: {model_name}")
	# Get the appropriate preprocess transform
	preprocess = preprocess_transforms[model_name]
	features_list = []
	# Extract features for all images
	for img_name, idx, cropped_image in image_data:
		input_tensor = preprocess(cropped_image)
		input_batch = input_tensor.unsqueeze(0).to(device)
		with torch.no_grad():
			features = model(input_batch)
			features = features.cpu().numpy().flatten()
			features_list.append(features)
	X = np.array(features_list)

	# Handle NaN or infinite values
	if np.isnan(X).any() or np.isinf(X).any():
		print("Data contains NaN or infinite values. Replacing them with zeros.")
		X = np.nan_to_num(X)

	# Optionally, reduce dimensionality with PCA
	pca = PCA(n_components=50)  # Adjust n_components as needed
	X_pca = pca.fit_transform(X)

	# Define output directory for the model
	model_output_dir = f"output_{model_name}"
	os.makedirs(model_output_dir, exist_ok=True)

	for algo_name, ClusteringAlgo in clustering_algorithms.items():
		print(f"\nClustering with {algo_name} algorithm")
		# Define output directory for the algorithm
		algo_output_dir = os.path.join(model_output_dir, algo_name)
		os.makedirs(algo_output_dir, exist_ok=True)
		requires_n_clusters = algo_name in ['kmeans', 'agglomerative', 'spectral']

		if requires_n_clusters:
			# Determine optimal number of clusters using Silhouette Score
			silhouette_scores = []
			cluster_range = range(2, 11)  # Testing number of clusters from 2 to 10

			for n_clusters in cluster_range:
				if algo_name == 'kmeans':
					clustering = ClusteringAlgo(n_clusters=n_clusters, random_state=42)
				elif algo_name == 'agglomerative':
					clustering = ClusteringAlgo(n_clusters=n_clusters)
				elif algo_name == 'spectral':
					clustering = ClusteringAlgo(n_clusters=n_clusters, affinity='nearest_neighbors',
												assign_labels='discretize', random_state=42)
				labels = clustering.fit_predict(X_pca)
				silhouette_avg = silhouette_score(X_pca, labels)
				silhouette_scores.append(silhouette_avg)

			# Plot and save the Silhouette Score graph
			plt.figure()
			plt.plot(cluster_range, silhouette_scores, 'bx-')
			plt.xlabel('Number of clusters')
			plt.ylabel('Silhouette Score')
			plt.title(f'Silhouette Analysis for {model_name} with {algo_name}')
			plt.savefig(os.path.join(algo_output_dir, f'silhouette_{model_name}_{algo_name}.png'))
			plt.close()

			# Choose the number of clusters based on the highest silhouette score
			optimal_n_clusters = cluster_range[np.argmax(silhouette_scores)]
			print(f"Optimal number of clusters for {model_name} with {algo_name}: {optimal_n_clusters}")

			# Perform clustering with optimal number of clusters
			if algo_name == 'kmeans':
				clustering = ClusteringAlgo(n_clusters=optimal_n_clusters, random_state=42)
			elif algo_name == 'agglomerative':
				clustering = ClusteringAlgo(n_clusters=optimal_n_clusters)
			elif algo_name == 'spectral':
				clustering = ClusteringAlgo(n_clusters=optimal_n_clusters, affinity='nearest_neighbors',
											assign_labels='discretize', random_state=42)
			labels = clustering.fit_predict(X_pca)

			# For K-Means, plot and save the elbow method graph
			if algo_name == 'kmeans':
				wcss = []
				for n_clusters in cluster_range:
					kmeans = KMeans(n_clusters=n_clusters, random_state=42)
					kmeans.fit(X_pca)
					wcss.append(kmeans.inertia_)
				plt.figure()
				plt.plot(cluster_range, wcss, 'bx-')
				plt.xlabel('Number of clusters')
				plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
				plt.title(f'Elbow Method for {model_name} with {algo_name}')
				plt.savefig(os.path.join(algo_output_dir, f'elbow_{model_name}_{algo_name}.png'))
				plt.close()
		else:
			# For DBSCAN, we'll try different eps values
			eps_values = [0.5, 1, 5, 10]
			silhouette_scores = []
			n_clusters_list = []
			for eps in eps_values:
				clustering = ClusteringAlgo(eps=eps, min_samples=5)
				labels = clustering.fit_predict(X_pca)
				n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
				n_clusters_list.append(n_clusters)
				if n_clusters > 1:
					silhouette_avg = silhouette_score(X_pca, labels)
					silhouette_scores.append(silhouette_avg)
				else:
					silhouette_scores.append(-1)  # Invalid silhouette score

			# Plot silhouette scores vs eps
			plt.figure()
			plt.plot(eps_values, silhouette_scores, 'bx-')
			plt.xlabel('eps value')
			plt.ylabel('Silhouette Score')
			plt.title(f'Silhouette Analysis for {model_name} with {algo_name}')
			plt.savefig(os.path.join(algo_output_dir, f'silhouette_{model_name}_{algo_name}.png'))
			plt.close()

			# Choose eps with highest silhouette score
			optimal_eps = eps_values[np.argmax(silhouette_scores)]
			print(f"Optimal eps for {model_name} with {algo_name}: {optimal_eps}")

			# Perform clustering with optimal eps
			clustering = ClusteringAlgo(eps=optimal_eps, min_samples=5)
			labels = clustering.fit_predict(X_pca)

		# Save clustering results
		cluster_dict = {}
		for label, (img_name, idx, cropped_image) in zip(labels, image_data):
			if label == -1:
				continue  # Ignore noise if any
			if label not in cluster_dict:
				cluster_dict[label] = []
			cluster_dict[label].append((img_name, idx, cropped_image))

		# Create combined images for each cluster
		for cluster_label, images_info in cluster_dict.items():
			num_images = len(images_info)
			# Compute the number of columns and rows to make collage as square as possible
			num_cols = int(math.ceil(math.sqrt(num_images)))
			num_rows = int(math.ceil(num_images / num_cols))
			thumbnail_size = 128
			collage_width = num_cols * thumbnail_size
			collage_height = num_rows * thumbnail_size

			collage_image = Image.new('RGB', (collage_width, collage_height), color=(255, 255, 255))

			for idx, (img_name, obj_idx, cropped_image) in enumerate(images_info):
				thumbnail = cropped_image.resize((thumbnail_size, thumbnail_size))
				# Paste the thumbnail into the collage
				row = idx // num_cols
				col = idx % num_cols
				collage_image.paste(thumbnail, (col * thumbnail_size, row * thumbnail_size))

			# Save the collage image
			collage_image.save(os.path.join(algo_output_dir, f'cluster_{cluster_label}.jpg'))

		# Save cluster assignments to a text file
		with open(os.path.join(algo_output_dir, "cluster_assignments.txt"), 'w') as f:
			for cluster_label, images_info in cluster_dict.items():
				f.write(f"Cluster {cluster_label}:\n")
				for img_name, obj_idx, _ in images_info:
					f.write(f"\t{img_name}_{obj_idx}\n")

		print(f"Results saved in directory: {algo_output_dir}")
