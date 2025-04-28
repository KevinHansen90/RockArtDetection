#!/usr/bin/env python3
from __future__ import annotations
import torch

from . import cpu, gpu


# ------------------------------------------------------------------ #
def build_train(is_detr: bool, device: torch.device, seed: int | None = None):
	"""Return a transform pipeline that lives **on the given device**."""
	if device.type in {"cpu", "mps"}:
		return cpu.build_train(is_detr, seed)
	if device.type == "cuda":
		return gpu.build_train(is_detr, device, seed)
	raise ValueError(f"Unsupported device type: {device.type}")


def build_val():
	"""Simple val/test pipeline (ToTensor + Normalize on CPU)."""
	return cpu.build_val()


# ------------------------------------------------------------------ #
# Helper so datasets can query which device a v2/Kornia pipeline lives on
# ------------------------------------------------------------------ #
def get_pipeline_device(pipeline):
	"""
    Return a torch.device if *pipeline* is the GPU TorchVision/Kornia pipeline,
    else ``None``.  Avoids importing torch in dataset code.
    """
	return getattr(pipeline, "_tv2_device", None)
