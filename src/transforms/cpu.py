#!/usr/bin/env python3
from __future__ import annotations
from typing import Sequence

import albumentations as A
import cv2
import torchvision.transforms as T
from albumentations.pytorch import ToTensorV2

# ── shared helpers ────────────────────────────────────────────────────────
_BASE_AUGS: list[A.BasicTransform] = [
	A.HorizontalFlip(p=0.5),
	A.RandomBrightnessContrast(0.1, 0.1, p=0.3),
	A.CLAHE(clip_limit=2.0, p=0.2),
	A.LongestMaxSize(max_size=1024, p=1.0),
]


def build_train(is_detr: bool, seed: int | None = None) -> A.Compose:  # noqa: D401
	ops: list[A.BasicTransform] = list(_BASE_AUGS)  # shallow copy

	# heavier geom. augments are fine on CPU
	ops.insert(1, A.Affine(translate_percent=0.1, scale=(0.85, 1.15), p=0.5))
	ops.insert(2, A.RandomRotate90(p=0.25))

	# ensure divisible by 32 for detector backbones
	ops.append(
		A.PadIfNeeded(
			min_height=None,
			min_width=None,
			pad_height_divisor=32,
			pad_width_divisor=32,
			border_mode=cv2.BORDER_CONSTANT,
			fill=0,
		)
	)
	ops.append(ToTensorV2())

	fmt = "yolo" if is_detr else "pascal_voc"
	return A.Compose(
		ops,
		bbox_params=A.BboxParams(
			format=fmt,
			label_fields=["class_labels"],
			min_visibility=0.4,
			check_each_transform=False,
			filter_invalid_bboxes=True,
		),
		seed=seed,
	)


# -------------------------------------------------------------------------
def build_val(
		mean: Sequence[float] | None = None,
		std: Sequence[float] | None = None,
) -> T.Compose:
	mean = mean or (0.485, 0.456, 0.406)
	std = std or (0.229, 0.224, 0.225)
	return T.Compose([T.ToTensor(), T.Normalize(mean, std)])
