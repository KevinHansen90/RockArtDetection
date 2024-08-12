import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Define simplified class labels
class_mapping = {
	"Zoomorfo (artiodactyla)": "Animal",
	"Zoomorfo (ave)": "Animal",
	"Zoomorfo (piche)": "Animal",
	"Zoomorfo (matuasto)": "Animal",
	"Antropomorfo": "Human",
	"Positivo de mano": "Hand",
	"Negativo de mano": "Hand",
	"Negativo de pata de choique": "Animal_print",
	"Negativo de puño": "Hand",
	"Círculos": "Geometric",
	"Círculos concéntricos": "Geometric",
	"Líneas rectas": "Geometric",
	"Líneas zigzag": "Geometric",
	"Escala": "Other",
	"Persona": "Human",
	"Lazo bola": "Other",
	"Conjuntos de puntos": "Geometric",
	"Impactos": "Other",
	"Tridígitos": "Animal_print"
}

# Create a mapping of simplified labels to numerical IDs
id2label = {i: label for i, label in enumerate(set(class_mapping.values()))}
label2id = {label: i for i, label in id2label.items()}


class AnimalRockArtDataset(Dataset):
	def __init__(self, image_dir, annotation_dir, processor):
		self.image_dir = image_dir
		self.annotation_dir = annotation_dir
		self.processor = processor
		self.image_files = []
		self.labels = []

		for img_file in os.listdir(image_dir):
			if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
				ann_file = os.path.join(annotation_dir, os.path.splitext(img_file)[0] + '.txt')
				if os.path.exists(ann_file):
					with open(ann_file, 'r') as f:
						for line in f:
							class_id = int(line.split()[0])
							original_class = list(class_mapping.keys())[class_id]
							simplified_class = class_mapping[original_class]
							if simplified_class == "Animal":
								self.image_files.append(img_file)
								self.labels.append(simplified_class)
								break

	def __len__(self):
		return len(self.image_files)

	def __getitem__(self, idx):
		img_path = os.path.join(self.image_dir, self.image_files[idx])
		image = Image.open(img_path).convert("RGB")
		image = image.resize((800, 800))
		inputs = self.processor(images=image, return_tensors="pt")
		return inputs.pixel_values.squeeze(), self.labels[idx]


# Set directories
image_dir = '../data/processed'
annotations_dir = '../data/annotations_processed'
output_dir = '../output/clustered_animals'

# Set up the model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
											   num_labels=len(id2label),
											   ignore_mismatched_sizes=True, revision="no_timm")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Create dataset and dataloader
dataset = AnimalRockArtDataset(image_dir, annotations_dir, processor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Extract features
features = []
filenames = []

with torch.no_grad():
	for images, _ in dataloader:
		images = images.to(device)
		outputs = model(pixel_values=images)
		batch_features = outputs.last_hidden_state.mean(dim=1)
		features.append(batch_features.cpu().numpy())
		filenames.extend([dataset.image_files[i] for i in range(len(_))])

features = np.concatenate(features)

# Normalize features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# Perform K-means clustering
n_clusters = 5  # You can adjust this number
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(normalized_features)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Visualize and save results
plt.figure(figsize=(10, 8))
for cluster in range(n_clusters):
	cluster_points = normalized_features[cluster_labels == cluster]
	plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')
plt.legend()
plt.title('Animal Rock Art Clusters')
plt.savefig(os.path.join(output_dir, 'clusters_visualization.png'))
plt.close()

# Save cluster assignments
with open(os.path.join(output_dir, 'cluster_assignments.txt'), 'w') as f:
	for filename, cluster in zip(filenames, cluster_labels):
		f.write(f"{filename}: Cluster {cluster}\n")

print(f"Clustering complete. Results saved to {output_dir}")


# Optional: Display a few examples from each cluster
def display_cluster_examples(cluster_id, num_examples=5):
	cluster_files = [f for f, c in zip(filenames, cluster_labels) if c == cluster_id]
	sample_files = np.random.choice(cluster_files, min(num_examples, len(cluster_files)), replace=False)

	fig, axes = plt.subplots(1, len(sample_files), figsize=(15, 3))
	for i, file in enumerate(sample_files):
		img = Image.open(os.path.join(image_dir, file)).convert('RGB')
		axes[i].imshow(img)
		axes[i].axis('off')
		axes[i].set_title(f"Cluster {cluster_id}")
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, f'cluster_{cluster_id}_examples.png'))
	plt.close()


for i in range(n_clusters):
	display_cluster_examples(i)

print("Example images from each cluster have been saved.")
