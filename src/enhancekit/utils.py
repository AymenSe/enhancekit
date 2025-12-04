"""Utility helpers for image loading, preprocessing, and postprocessing."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, Image.Image, np.ndarray, torch.Tensor]


def load_image(source: ImageInput) -> Image.Image:
    """Load an image from various input types into a PIL Image."""

    if isinstance(source, Image.Image):
        return source
    if isinstance(source, torch.Tensor):
        array = source.detach().cpu().numpy()
        if array.ndim == 3:
            array = np.transpose(array, (1, 2, 0))
        return Image.fromarray(np.clip(array * 255.0, 0, 255).astype("uint8"))
    if isinstance(source, np.ndarray):
        array = source
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        return Image.fromarray(array.astype("uint8"))
    return Image.open(source).convert("RGB")


def to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a float tensor in CHW format scaled to [0, 1]."""

    array = np.array(image).astype("float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor


def from_tensor(tensor: torch.Tensor) -> Image.Image:
    """Convert a tensor in CHW format back to a PIL Image."""

    tensor = tensor.detach().cpu().clamp(0, 1)
    array = tensor.permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255.0).astype("uint8"))


def normalize(tensor: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    """Apply channel-wise normalization."""

    mean_tensor = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def denormalize(tensor: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    """Invert channel-wise normalization."""

    mean_tensor = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return tensor * std_tensor + mean_tensor


def resolve_images_from_folder(folder: Union[str, Path]) -> List[Path]:
    """Return sorted list of image files in a folder."""

    path = Path(folder)
    images = sorted([p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
    logger.debug("Resolved %d images from %s", len(images), path)
    return images


__all__ = [
    "load_image",
    "to_tensor",
    "from_tensor",
    "normalize",
    "denormalize",
    "resolve_images_from_folder",
    "ImageInput",
]
