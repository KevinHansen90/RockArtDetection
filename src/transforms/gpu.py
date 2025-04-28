#!/usr/bin/env python3
from __future__ import annotations
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as v2

# Optional Kornia fallback ---------------------------------------------------
try:
    import kornia.augmentation as K  # type: ignore
except ImportError:  # pragma: no cover
    K = None

# --------------------------------------------------------------------------- #
# PadToMultiple shim for torchvision 0.17.x
# --------------------------------------------------------------------------- #
if hasattr(v2, "PadToMultiple"):
    _PadToMultiple = v2.PadToMultiple
else:

    class _PadToMultiple(torch.nn.Module):
        """Pad right / bottom so H, W become multiples of *divisor*."""
        def __init__(self, divisor: int = 32, fill: int = 0) -> None:
            super().__init__()
            self.divisor = divisor
            self.fill = fill

        def forward(self, sample):
            img = sample["image"]
            h, w = img.shape[-2:]
            pad_h = (self.divisor - h % self.divisor) % self.divisor
            pad_w = (self.divisor - w % self.divisor) % self.divisor
            if pad_h or pad_w:
                sample["image"] = F.pad(img, (0, pad_w, 0, pad_h), value=self.fill)
            return sample


# --------------------------------------------------------------------------- #
def _tv2_train(is_detr: bool) -> v2.Compose:
    box_fmt = "cxcywh" if is_detr else "xyxy"
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(translate=(0.1, 0.1), scale=(0.85, 1.15), degrees=0),
            v2.ColorJitter(0.1, 0.1, 0.1, 0.1),
            _PadToMultiple(32, fill=0),
            v2.Normalize(),
        ],
        bbox_format=box_fmt,
        antialias=True,
    )


def _kornia_train() -> "K.ImageSequential":  # type: ignore[name-defined]
    return K.ImageSequential(
        K.RandomHorizontalFlip(),
        K.RandomAffine(degrees=0, translate=0.1, scale=(0.85, 1.15)),
        K.ColorJitter(0.1, 0.1, 0.1, 0.1),
        K.PadTo(size_divisor=32, pad_mode="constant"),
        data_key="input",
        keepdim=True,
    )


# --------------------------------------------------------------------------- #
def build_train(is_detr: bool, device: torch.device, seed: int | None = None):
    """Return a GPU-resident transform pipeline."""
    torch_rng = torch.Generator(device=device)
    if seed is not None:
        torch_rng.manual_seed(seed)

    if device.type in {"cuda", "mps"}:
        pipe = _tv2_train(is_detr)
    elif device.type == "cuda" and K is not None:
        pipe = _kornia_train()
    else:
        raise RuntimeError("GPU transform requested but no suitable backend found.")

    pipe.set_rng_state(torch_rng.get_state())
    pipe = pipe.to(device)            # move op params to device
    pipe._tv2_device = device         # <- tag so dataset can query
    return pipe
