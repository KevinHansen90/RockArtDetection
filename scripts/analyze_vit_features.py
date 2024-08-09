import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load the features
def load_features(file_path):
    return np.load(file_path)

# Visualize the features using a heatmap
def visualize_heatmap(features, output_path):
    plt.figure(figsize=(10, 10))
    sns.heatmap(features.squeeze(), cmap='viridis')
    plt.title("Feature Heatmap")
    plt.savefig(output_path)
    plt.close()

# Visualize the features using PCA
def visualize_pca(features, output_path):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features.squeeze())
    plt.figure(figsize=(10, 10))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], s=5, cmap='viridis')
    plt.title("PCA of Features")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig(output_path)
    plt.close()

# Visualize the features using t-SNE
def visualize_tsne(features, output_path):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    reduced_features = tsne.fit_transform(features.squeeze())
    plt.figure(figsize=(10, 10))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], s=5, cmap='viridis')
    plt.title("t-SNE of Features")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig(output_path)
    plt.close()

# Directories
features_dir = '../output/features'
output_dir = '../output/feature_analysis'

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Analyze the features
for idx, filename in enumerate(os.listdir(features_dir)):
    if filename.endswith(".npy"):
        file_path = os.path.join(features_dir, filename)
        features = load_features(file_path)

        # Visualize and save the heatmap
        heatmap_path = os.path.join(output_dir, f"heatmap_{idx + 1}.png")
        visualize_heatmap(features, heatmap_path)
        print(f"Saved heatmap to {heatmap_path}")

        # Visualize and save the PCA
        pca_path = os.path.join(output_dir, f"pca_{idx + 1}.png")
        visualize_pca(features, pca_path)
        print(f"Saved PCA to {pca_path}")

        # Visualize and save the t-SNE
        tsne_path = os.path.join(output_dir, f"tsne_{idx + 1}.png")
        visualize_tsne(features, tsne_path)
        print(f"Saved t-SNE to {tsne_path}")

print("Feature analysis complete.")