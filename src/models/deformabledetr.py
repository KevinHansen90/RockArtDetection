import torch
from transformers import DeformableDetrForObjectDetection


class DeformableDETRModel(torch.nn.Module):
	def __init__(self, num_classes):
		super(DeformableDETRModel, self).__init__()
		self.model = DeformableDetrForObjectDetection.from_pretrained(
			"SenseTime/deformable-detr",
			num_labels=num_classes,
			ignore_mismatched_sizes=True
		)

	def forward(self, pixel_values, pixel_mask=None, labels=None):
		return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

	def train(self, mode=True):
		self.training = mode
		for module in self.children():
			module.train(mode)
		return self
