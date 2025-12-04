"""Core model abstractions for enhancekit."""
from __future__ import annotations

import logging
from urllib.parse import urlparse
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from torch import nn

from .config import FreezeMode, ModelConfig, Precision
from .utils import ImageInput, from_tensor, load_image, resolve_images_from_folder, to_tensor

logger = logging.getLogger(__name__)


class BaseEnhancementModel(nn.Module):
    """Base class for image enhancement models."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.device_type = config.device
        self.precision = config.precision
        self._setup_precision()

    def _setup_precision(self) -> None:
        if self.precision == Precision.FP16:
            self.half()
        elif self.precision == Precision.BF16:
            self.bfloat16()

    def load_weights(self, checkpoint: Optional[str] = None) -> None:
        path = checkpoint or self.config.checkpoint
        if not path:
            logger.info("No checkpoint provided for %s", self.config.name)
            return
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            ckpt_path = self._download_checkpoint(path)
        if ckpt_path is None or not ckpt_path.exists():
            repo_hint = f" in repository {self.config.repository}" if self.config.repository else ""
            logger.warning("Checkpoint %s does not exist%s; skipping load", path, repo_hint)
            return
        state = torch.load(ckpt_path, map_location="cpu")
        state_dict = state.get("state_dict", state)
        self.load_state_dict(state_dict, strict=False)
        logger.info("Loaded weights for %s from %s", self.config.name, ckpt_path)

    def _download_checkpoint(self, location: str) -> Optional[Path]:
        """Download a checkpoint to the local cache if a URL is available."""

        url = self.config.download_url or location
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return None
        cache_dir = Path(torch.hub.get_dir()) / "enhancekit"
        cache_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(parsed.path).name or f"{self.config.name}.pth"
        cache_path = cache_dir / filename
        if cache_path.exists():
            logger.info("Using cached checkpoint for %s at %s", self.config.name, cache_path)
            return cache_path
        try:
            torch.hub.download_url_to_file(url, cache_path)
        except Exception as exc:  # pragma: no cover - network-dependent
            logger.warning("Failed to download checkpoint from %s: %s", url, exc)
            return None
        logger.info("Downloaded checkpoint for %s to %s", self.config.name, cache_path)
        return cache_path

    def to(self, *args, **kwargs):  # type: ignore[override]
        module = super().to(*args, **kwargs)
        self.device_type = str(args[0]) if args else kwargs.get("device", self.device_type)
        return module

    def freeze(self, mode: Optional[FreezeMode] = None) -> None:
        if mode is False:  # type: ignore[comparison-overlap]
            return
        if mode is True:
            freeze_mode = FreezeMode.ALL
        else:
            freeze_mode = self.config.freeze_mode if mode is None else mode
        if freeze_mode == FreezeMode.ALL:
            for param in self.parameters():
                param.requires_grad = False
        elif freeze_mode == FreezeMode.BACKBONE:
            backbone = getattr(self, "backbone", None)
            if backbone is None:
                logger.warning("Backbone freezing requested but no backbone attribute found")
            else:
                for param in backbone.parameters():
                    param.requires_grad = False
        logger.info("Applied freeze mode %s to %s", freeze_mode.value, self.config.name)

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Unfroze all parameters for %s", self.config.name)

    def enhance_image(self, image: ImageInput) -> ImageInput:
        self.eval()
        pil_image = load_image(image)
        tensor = to_tensor(pil_image).unsqueeze(0).to(self.device_type)
        with torch.inference_mode():
            output = self.forward(tensor)[0]
        return from_tensor(output)

    def enhance_batch(self, images: Iterable[ImageInput]) -> List[ImageInput]:
        outputs: List[ImageInput] = []
        self.eval()
        batch_tensors = []
        for img in images:
            tensor = to_tensor(load_image(img))
            batch_tensors.append(tensor)
        if not batch_tensors:
            return outputs
        batch = torch.stack(batch_tensors, dim=0).to(self.device_type)
        with torch.inference_mode():
            results = self.forward(batch)
        for tensor in results:
            outputs.append(from_tensor(tensor))
        return outputs

    def enhance_folder(self, folder: str, output_folder: Optional[str] = None) -> List[Path]:
        images = resolve_images_from_folder(folder)
        outputs: List[Path] = []
        target_dir = Path(output_folder) if output_folder else None
        if target_dir:
            target_dir.mkdir(parents=True, exist_ok=True)
        for path in images:
            enhanced = self.enhance_image(path)
            if isinstance(enhanced, Path):
                outputs.append(enhanced)
            else:
                if target_dir:
                    output_path = target_dir / path.name
                    enhanced.save(output_path)
                    outputs.append(output_path)
        return outputs


class IdentityBackbone(nn.Module):
    """A minimal backbone used as a placeholder for documentation and tests."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x


class ExampleEnhancementModel(BaseEnhancementModel):
    """Simple example model performing identity mapping with optional gain."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.gain = nn.Parameter(torch.tensor(config.kwargs.get("gain", 1.0), dtype=torch.float32))
        self.backbone = IdentityBackbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        y = self.backbone(x)
        return y * self.gain


__all__ = [
    "BaseEnhancementModel",
    "ExampleEnhancementModel",
    "IdentityBackbone",
]
