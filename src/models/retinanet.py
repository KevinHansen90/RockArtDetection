import torch
from torchvision.models.detection import retinanet_resnet50_fpn


class RetinaNetModel(torch.nn.Module):
	def __init__(self, num_classes):
		super(RetinaNetModel, self).__init__()
		self.model = retinanet_resnet50_fpn(weights='DEFAULT')

		# Modify the classification head for the new number of classes
		num_anchors = self.model.head.classification_head.num_anchors
		in_channels = 256  # This is the number of input channels for the classification head

		self.model.head.classification_head.num_classes = num_classes
		self.model.head.classification_head.cls_logits = torch.nn.Conv2d(
			in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
		)

	def forward(self, images, targets=None):
		if self.training and targets is None:
			raise ValueError("In training mode, targets should be passed")
		return self.model(images, targets)

	def train(self, mode=True):
		self.training = mode
		for module in self.children():
			module.train(mode)
		return self
