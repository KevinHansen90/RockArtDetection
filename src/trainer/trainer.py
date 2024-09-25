import torch
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision
from evaluation.validation import calculate_retinanet_losses, eval_forward


class Trainer:
	def __init__(self, model, optimizer, device, model_type):
		self.model = model
		self.optimizer = optimizer
		self.device = device
		self.model_type = model_type

	def train_one_epoch(self, data_loader):
		self.model.train()
		total_loss = 0.0

		for batch in tqdm(data_loader, desc="Training", leave=True):
			self.optimizer.zero_grad()

			if self.model_type == 'deformable_detr':
				pixel_values = batch["pixel_values"].to(self.device)
				labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
				outputs = self.model(pixel_values=pixel_values, labels=labels)
				loss_dict = outputs.loss_dict
			else:  # Faster R-CNN and RetinaNet
				pixel_values, labels = batch
				images = list(image.to(self.device) for image in pixel_values)
				targets = [{k: v.to(self.device) for k, v in label.items()} for label in labels]
				loss_dict = self.model(images, targets)

			losses = sum(loss for loss in loss_dict.values())

			losses.backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
			self.optimizer.step()

			total_loss += losses.item()

		avg_loss = total_loss / len(data_loader)
		return avg_loss

	def evaluate(self, data_loader):
		self.model.eval()

		map_metric = MeanAveragePrecision(class_metrics=True).to(self.device)

		all_losses = []

		with torch.inference_mode():
			for batch in tqdm(data_loader, desc="Evaluating"):
				if self.model_type == 'deformable_detr':
					pixel_values = batch["pixel_values"].to(self.device)
					labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
					outputs = self.model(pixel_values=pixel_values, labels=labels)
					loss_dict = outputs.loss_dict
					predictions = self.model(pixel_values=pixel_values)
					pred_boxes = predictions.pred_boxes
					logits = predictions.logits
					detections = [
						{'boxes': pred_box, 'scores': torch.max(logit, dim=-1)[0],
						 'labels': torch.argmax(logit, dim=-1)}
						for pred_box, logit in zip(pred_boxes, logits)
					]
					for label in labels:
						if 'class_labels' in label:
							label['labels'] = label.pop('class_labels')
				else:  # Faster R-CNN and RetinaNet
					images = list(img.to(self.device) for img in batch[0])
					targets = [{k: v.to(self.device) for k, v in t.items()} for t in batch[1]]

					if self.model_type == 'retinanet':
						# Get predictions
						predictions = self.model(images)

						# Calculate losses specific to RetinaNet
						# cls_loss, box_loss = calculate_retinanet_losses(predictions, targets)
						losses, detections = calculate_retinanet_losses(self.model, images, targets)
						cls_loss = losses['classification']
						box_loss = losses['bbox_regression']
						loss_dict = {'cls_loss': cls_loss.item(), 'box_loss': box_loss.item()}

						# Process predictions for mAP metric
						detections = []
						for pred in predictions:
							detection = {
								'boxes': pred['boxes'],
								'scores': pred['scores'],
								'labels': pred['labels']
							}
							detections.append(detection)

					else:  # Faster R-CNN
						# Use eval_forward for Faster R-CNN
						losses, detections = eval_forward(self.model, images, targets)
						loss_dict = {k: v.item() for k, v in losses.items()}

				map_metric.update(detections, labels if self.model_type == 'deformable_detr' else targets)
				all_losses.append({k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()})

		avg_losses = {k: sum(d[k] for d in all_losses) / len(all_losses) for k in all_losses[0]}
		map_results = map_metric.compute()

		return {
			'losses': avg_losses,
			'mAP': map_results['map'].item(),
			'mAP_50': map_results['map_50'].item(),
			'mAP_75': map_results['map_75'].item(),
			'mAP_small': map_results['map_small'].item(),
			'mAP_medium': map_results['map_medium'].item(),
			'mAP_large': map_results['map_large'].item(),
			'mar_1': map_results['mar_1'].item(),
			'mar_10': map_results['mar_10'].item(),
			'mar_100': map_results['mar_100'].item(),
			'map_per_class': map_results['map_per_class'].tolist() if 'map_per_class' in map_results else None,
			'mar_100_per_class': map_results[
				'mar_100_per_class'].tolist() if 'mar_100_per_class' in map_results else None
		}
