import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNNModel(torch.nn.Module):
	def __init__(self, num_classes):
		super(FasterRCNNModel, self).__init__()
		self.model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
		in_features = self.model.roi_heads.box_predictor.cls_score.in_features
		self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

		# Expose necessary components
		self.transform = self.model.transform
		self.backbone = self.model.backbone
		self.rpn = self.model.rpn
		self.roi_heads = self.model.roi_heads

	def forward(self, images, targets=None):
		if self.training and targets is None:
			raise ValueError("In training mode, targets should be passed")
		return self.model(images, targets)

	def train(self, mode=True):
		self.training = mode
		for module in self.children():
			module.train(mode)
		return self
