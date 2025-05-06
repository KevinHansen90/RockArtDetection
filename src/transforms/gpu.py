#!/usr/bin/env python3
from __future__ import annotations

import inspect
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as v2

# ----------------------------- optional Kornia ----------------------------- #
try:
    import kornia.augmentation as K  # type: ignore
except ImportError:  # pragma: no cover
    K = None

# ----------------------------- constants ----------------------------------- #
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------- PadToMultiple shim for torchvision < 0.18 ---------------- #
if hasattr(v2, "PadToMultiple"):
    _PadToMultiple = v2.PadToMultiple
else:

    class _PadToMultiple(torch.nn.Module):
        """Pad so H & W become multiples of *divisor* (right / bottom only)."""
        def __init__(self, divisor: int = 32, fill: int = 0) -> None:
            super().__init__()
            self.divisor, self.fill = divisor, fill

        def forward(self, sample):
            img = sample["image"]
            h, w = img.shape[-2:]
            pad_h = (self.divisor - h % self.divisor) % self.divisor
            pad_w = (self.divisor - w % self.divisor) % self.divisor
            if pad_h or pad_w:
                sample["image"] = F.pad(img, (0, pad_w, 0, pad_h), value=self.fill)
            return sample


# -------------------------- helper: safe Compose --------------------------- #
_COMPOSE_SIG = inspect.signature(v2.Compose)
_SUPPORTS_BBOX_FORMAT = "bbox_format" in _COMPOSE_SIG.parameters
_SUPPORTS_ANTIALIAS = "antialias" in _COMPOSE_SIG.parameters


def _tv2_train(is_detr: bool) -> v2.Compose:
    """Torchvision-v2 GPU pipeline, robust across 0.15 → 0.18."""
    transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(translate=(0.1, 0.1), scale=(0.85, 1.15), degrees=0),
        v2.ColorJitter(0.1, 0.1, 0.1, 0.1),
        _PadToMultiple(32, fill=0),
        v2.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]

    kwargs: dict[str, object] = {}
    if _SUPPORTS_ANTIALIAS:
        kwargs["antialias"] = True
    if _SUPPORTS_BBOX_FORMAT:
        kwargs["bbox_format"] = "cxcywh" if is_detr else "xyxy"

    return v2.Compose(transforms, **kwargs)  # type: ignore[arg-type]


# ----------------------------- Kornia branch -------------------------------- #
def _kornia_train() -> "K.ImageSequential":  # type: ignore[name-defined]
    """GPU-only alternative when Kornia is installed."""
    return K.ImageSequential(
        K.RandomHorizontalFlip(),
        K.RandomAffine(degrees=0, translate=0.1, scale=(0.85, 1.15)),
        K.ColorJitter(0.1, 0.1, 0.1, 0.1),
        K.PadTo(size_divisor=32, pad_mode="constant"),
        data_key="input",
        keepdim=True,
    )


# -------------------------- public build_train ----------------------------- #
def build_train(is_detr: bool, device: torch.device, seed: int | None = None):
    """Return a transform pipeline that already lives on *device*."""
    torch_rng = torch.Generator(device=device)
    if seed is not None:
        torch_rng.manual_seed(seed)

    pipe = _kornia_train() if device.type == "cuda" and K is not None else _tv2_train(is_detr)

    if hasattr(pipe, "set_rng_state"):  # only torchvision pipelines expose this
        pipe.set_rng_state(torch_rng.get_state())

    pipe = pipe.to(device)                        # move params to GPU / MPS
    pipe._tv2_device = device                     # tag for dataset introspection
    return pipe