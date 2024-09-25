import torch
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers
from collections import OrderedDict


def calculate_retinanet_losses(model, images, targets):
	# Get the original image sizes
	original_image_sizes = []
	for img in images:
		val = img.shape[-2:]
		original_image_sizes.append((val[0], val[1]))

	# Transform the input
	images, targets = model.model.transform(images, targets)

	# Get the features from the backbone
	features = model.model.backbone(images.tensors)
	if isinstance(features, torch.Tensor):
		features = OrderedDict([("0", features)])
	features = list(features.values())

	# Compute the RetinaNet heads outputs using the features
	head_outputs = model.model.head(features)

	# Create the set of anchors
	anchors = model.model.anchor_generator(images, features)

	# Compute losses
	losses = model.model.compute_loss(targets, head_outputs, anchors)

	# Compute detections
	try:
		detections = model.model.postprocess_detections(head_outputs, anchors, images.image_sizes)
		detections = model.model.transform.postprocess(detections, images.image_sizes, original_image_sizes)
	except IndexError:
		# If no detections are made, return empty lists
		detections = [{'boxes': torch.empty((0, 4), device=images.tensors.device),
					   'labels': torch.empty((0,), device=images.tensors.device, dtype=torch.long),
					   'scores': torch.empty((0,), device=images.tensors.device)}
					  for _ in range(len(images.tensors))]

	return losses, detections


def eval_forward(model, images, targets):
	original_image_sizes = []
	for img in images:
		val = img.shape[-2:]
		original_image_sizes.append((val[0], val[1]))

	images, targets = model.transform(images, targets)

	# Check for degenerate boxes
	if targets is not None:
		for target_idx, target in enumerate(targets):
			boxes = target["boxes"]
			degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
			if degenerate_boxes.any():
				bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
				degen_bb = boxes[bb_idx].tolist()
				raise ValueError(
					"All bounding boxes should have positive height and width."
					f" Found invalid box {degen_bb} for target at index {target_idx}."
				)

	features = model.backbone(images.tensors)
	if isinstance(features, torch.Tensor):
		features = OrderedDict([("0", features)])

	model.rpn.training = True
	features_rpn = list(features.values())
	objectness, pred_bbox_deltas = model.rpn.head(features_rpn)
	anchors = model.rpn.anchor_generator(images, features_rpn)

	num_images = len(anchors)
	num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
	num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
	objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)

	proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
	proposals = proposals.view(num_images, -1, 4)
	proposals, scores = model.rpn.filter_proposals(proposals, objectness, images.image_sizes,
												   num_anchors_per_level)

	proposal_losses = {}
	assert targets is not None
	labels, matched_gt_boxes = model.rpn.assign_targets_to_anchors(anchors, targets)
	regression_targets = model.rpn.box_coder.encode(matched_gt_boxes, anchors)
	loss_objectness, loss_rpn_box_reg = model.rpn.compute_loss(
		objectness, pred_bbox_deltas, labels, regression_targets
	)
	proposal_losses = {
		"loss_objectness": loss_objectness,
		"loss_rpn_box_reg": loss_rpn_box_reg,
	}

	proposals, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(proposals,
																								  targets)
	box_features = model.roi_heads.box_roi_pool(features, proposals, images.image_sizes)
	box_features = model.roi_heads.box_head(box_features)
	class_logits, box_regression = model.roi_heads.box_predictor(box_features)

	result = []
	detector_losses = {}
	loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
	detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
	boxes, scores, labels = model.roi_heads.postprocess_detections(class_logits, box_regression, proposals,
																   images.image_sizes)
	num_images = len(boxes)
	for i in range(num_images):
		result.append(
			{
				"boxes": boxes[i],
				"labels": labels[i],
				"scores": scores[i],
			}
		)
	detections = result
	detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)

	model.rpn.training = False
	model.roi_heads.training = False
	losses = {}
	losses.update(detector_losses)
	losses.update(proposal_losses)

	return losses, detections


# VER SI SE PUEDE CON TORCHMETRICS
# TORCHINFO -SUMMARY para ver modelo