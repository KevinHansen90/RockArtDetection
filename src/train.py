import csv
import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from utils.utils import set_seed, load_model, load_optimizer, load_scheduler, collate_fn, collate_fn_detr, setup_logger
from trainer.trainer import Trainer
from data.dataset import RockArtDataset
from data.transforms import get_transforms


def main(config_path):
	# Load configuration
	with open(config_path, 'r') as file:
		config = yaml.safe_load(file)

	# Setup logger
	log_dir = config['checkpoint_dir']
	logger = setup_logger(log_dir)

	# Create CSV file for metrics
	csv_path = os.path.join(log_dir, 'metrics.csv')
	csv_file = open(csv_path, 'w', newline='')
	csv_writer = csv.writer(csv_file)
	csv_writer.writerow(['Epoch', 'Train Loss'])

	# Set random seed for reproducibility
	set_seed(config['seed'])

	# Set device
	device = torch.device(config['device'])

	# Load transformations
	train_transforms = get_transforms(
		config['train_transforms'],
		train=True,
		image_size=config.get('image_size', (512, 512))
	) if config['train_transforms'] else None

	val_transforms = get_transforms(
		config['val_transforms'],
		train=False,
		image_size=config.get('image_size', (512, 512))
	) if config['val_transforms'] else None

	# Load datasets
	train_dataset = RockArtDataset(config['train_dir'], transforms=train_transforms,
								   model_type=config['model']['type'])
	val_dataset = RockArtDataset(config['val_dir'], transforms=val_transforms,
								 model_type=config['model']['type'])

	# Create data loaders
	if config['model']['type'] == 'deformable_detr':
		train_loader = DataLoader(train_dataset, collate_fn=collate_fn_detr, **config['train_dataloader'])
		val_loader = DataLoader(val_dataset, collate_fn=collate_fn_detr, **config['val_dataloader'])
	else:
		train_loader = DataLoader(train_dataset, collate_fn=collate_fn, **config['train_dataloader'])
		val_loader = DataLoader(val_dataset, collate_fn=collate_fn, **config['val_dataloader'])

	# Load model
	model = load_model(config['model'])
	model.to(device)

	# Load optimizer
	optimizer = load_optimizer(model, config['optimizer'])

	# Load scheduler (if specified)
	scheduler = load_scheduler(optimizer, config['scheduler']) if 'scheduler' in config else None

	# Initialize trainer
	trainer = Trainer(model, optimizer, device, config['model']['type'])

	# Training loop
	best_loss = float('inf')
	for epoch in range(config['num_epochs']):
		logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}")

		train_loss = trainer.train_one_epoch(train_loader)
		logger.info(f"Training Loss: {train_loss:.4f}")

		eval_metrics = trainer.evaluate(val_loader)
		logger.info("Validation Metrics:")
		for k, v in eval_metrics.items():
			if isinstance(v, (float, int)):
				logger.info(f"{k}: {v:.4f}")
			else:
				logger.info(f"{k}: {v}")

		if scheduler:
			scheduler.step()

		# Save metrics to CSV
		if epoch == 0:
			csv_file.seek(0)  # Go to the beginning of the file
			csv_writer.writerow(['Epoch', 'Train Loss'] + list(eval_metrics.keys()))

		# Write the metrics
		csv_writer.writerow([epoch + 1, train_loss] + list(eval_metrics.values()))

		# Save best and last model
		if train_loss < best_loss:
			best_loss = train_loss
			torch.save(model.state_dict(), f"{config['checkpoint_dir']}/best.pth")
			logger.info(f"Saved new best model with loss: {best_loss:.4f}")

		torch.save({
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
		}, f"{config['checkpoint_dir']}/last.pth")

	# Close CSV file
	csv_file.close()

	logger.info("Training completed.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Fine-tune object detection model")
	parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
	args = parser.parse_args()
	main(args.config)

	# python train.py --config configs/config_faster_rcnn.yaml
