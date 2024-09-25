import os
import random
import numpy as np
import torch
from models.fasterrcnn import FasterRCNNModel
from models.retinanet import RetinaNetModel
from models.deformabledetr import DeformableDETRModel
import torchvision.transforms.functional as F
import logging


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)


def load_model(model_config):
	if model_config['type'] == 'faster_rcnn':
		model = FasterRCNNModel(num_classes=model_config['num_classes'])
	elif model_config['type'] == 'retinanet':
		model = RetinaNetModel(num_classes=model_config['num_classes'])
	elif model_config['type'] == 'deformable_detr':
		model = DeformableDETRModel(num_classes=model_config['num_classes'])
	else:
		raise ValueError(f"Unsupported model type: {model_config['type']}")
	return model


def load_optimizer(model, optimizer_config):
	if optimizer_config['type'] == 'Adam':
		return torch.optim.Adam(model.parameters(), lr=optimizer_config['lr'])
	elif optimizer_config['type'] == 'AdamW':
		return torch.optim.AdamW(model.parameters(), lr=optimizer_config['lr'],
								 weight_decay=optimizer_config['weight_decay'])
	else:
		raise ValueError(f"Unsupported optimizer type: {optimizer_config['type']}")


def load_scheduler(optimizer, scheduler_config):
	if scheduler_config['type'] == 'StepLR':
		return torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_config['step_size'],
											   gamma=scheduler_config['gamma'])
	elif scheduler_config['type'] == 'ReduceLROnPlateau':
		return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_config['mode'],
														  factor=scheduler_config['factor'],
														  patience=scheduler_config['patience'])
	elif scheduler_config['type'] == 'CosineAnnealingLR':
		return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_config['T_max'])
	else:
		raise ValueError(f"Unsupported scheduler type: {scheduler_config['type']}")


def collate_fn(batch):
	return tuple(zip(*batch))


def collate_fn_detr(batch):
	pixel_values = []
	labels = []

	for item in batch:
		pixel_values.append(item["pixel_values"])
		labels.append({k: v.to(pixel_values[-1].device) for k, v in item["labels"].items()})

	# Find the maximum dimensions
	max_h = max([img.shape[1] for img in pixel_values])
	max_w = max([img.shape[2] for img in pixel_values])

	# Pad images to the maximum size
	padded_pixel_values = []
	for img in pixel_values:
		pad_h = max_h - img.shape[1]
		pad_w = max_w - img.shape[2]
		padded_img = F.pad(img, (0, pad_w, 0, pad_h))
		padded_pixel_values.append(padded_img)

	# Stack the padded images
	pixel_values = torch.stack(padded_pixel_values)

	encoding = {
		"pixel_values": pixel_values,
		"labels": labels
	}
	return encoding


def setup_logger(log_dir):
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.INFO)

	# Create handlers
	c_handler = logging.StreamHandler()
	f_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
	c_handler.setLevel(logging.INFO)
	f_handler.setLevel(logging.INFO)

	# Create formatters and add it to handlers
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	c_handler.setFormatter(formatter)
	f_handler.setFormatter(formatter)

	# Add handlers to the logger
	logger.addHandler(c_handler)
	logger.addHandler(f_handler)

	return logger
