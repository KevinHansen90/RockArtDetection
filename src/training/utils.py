# src/training/utils.py

import csv
import yaml
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as T
from tqdm import tqdm


def load_config(path):
	"""Reads a YAML config file."""
	with open(path, "r") as f:
		return yaml.safe_load(f)


def get_device():
	"""
    Return the best available device:
      - MPS (Apple Silicon) if available
      - otherwise CUDA if available
      - otherwise CPU
    """
	if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		return torch.device("mps")
	elif torch.cuda.is_available():
		return torch.device("cuda")
	else:
		return torch.device("cpu")


def get_simple_transform():
	"""No custom filter or augmentation, just convert PIL to Tensor."""

	return T.Compose([
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406],
					std=[0.229, 0.224, 0.225])
	])


def plot_curve(values, ylabel, title, output_path):
	plt.figure(figsize=(8, 6))
	plt.plot(range(1, len(values) + 1), values, marker="o")
	plt.xlabel("Epoch")
	plt.ylabel(ylabel)
	plt.title(title)
	plt.grid(True)
	plt.savefig(output_path)
	plt.close()
	tqdm.write(f"Saved {title} plot: {output_path}")


def save_metrics_csv(csv_path, train_losses, val_losses, map_list, mar_list):
	"""Save epoch-level metrics to a CSV file."""
	with open(csv_path, mode="w", newline="") as f:
		writer = csv.writer(f)
		# Write header
		writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "mAP@0.5", "mAR@100"])
		# Write data
		for epoch, (train_loss, val_loss, map_val, mar_val) in enumerate(
				zip(train_losses, val_losses, map_list, mar_list), start=1):
			writer.writerow([epoch, train_loss, val_loss, map_val, mar_val])


def freeze_batchnorm(model):
	"""Freeze backbone BN and unfreeze head BN with logging."""
	# Freeze backbone BN and log which modules are frozen.
	for name, module in model.backbone.named_modules():
		if isinstance(module, nn.BatchNorm2d):
			module.eval()
			for param in module.parameters():
				param.requires_grad = False

	def unfreeze_bn(module, prefix=""):
		if isinstance(module, nn.BatchNorm2d):
			module.train()
			for param in module.parameters():
				param.requires_grad = True
		for name, child in module.named_children():
			unfreeze_bn(child, prefix=f"{prefix}{name}/")

	# Apply to the head.
	model.head.apply(lambda m: unfreeze_bn(m))
	return model


def compute_total_loss(loss_dict):
	if isinstance(loss_dict, dict):
		return sum(loss.sum() for loss in loss_dict.values())
	return loss_dict.loss
