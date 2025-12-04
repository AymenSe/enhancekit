"""Configuration definitions for enhancekit models."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

# Optional dependency: attempt to import PyYAML, otherwise fall back to JSON parsing
try:  # pragma: no cover - exercised implicitly during import
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - fallback only when dependency missing
    import json as yaml  # type: ignore
    logger.warning(
        "PyYAML is not installed; YAML configs will be parsed as JSON-compatible structures."
    )


class Precision(Enum):
    """Supported precision policies for model execution."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class FreezeMode(Enum):
    """Different strategies for freezing model parameters."""

    NONE = "none"
    BACKBONE = "backbone"
    ALL = "all"


@dataclass
class ModelConfig:
    """Dataclass capturing model configuration.

    Parameters
    ----------
    name:
        Unique model identifier.
    task:
        Task category (e.g., "super-resolution", "denoising").
    architecture:
        Human-readable architecture description (e.g., "SwinIR x4").
    checkpoint:
        Optional path to pretrained weights.
    device:
        Default device for the model ("cpu" or "cuda").
    precision:
        Precision policy for inference/training.
    freeze_mode:
        Strategy for freezing parameters on load.
    kwargs:
        Additional keyword arguments forwarded to the underlying model
        builder. These can include hyperparameters such as upscale factor
        or number of blocks.
    """

    name: str
    task: str
    architecture: str
    checkpoint: Optional[str] = None
    device: str = "cpu"
    precision: Precision = Precision.FP32
    freeze_mode: FreezeMode = FreezeMode.NONE
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create a ``ModelConfig`` instance from a mapping."""

        precision_value = data.get("precision", Precision.FP32)
        precision = Precision(precision_value) if not isinstance(precision_value, Precision) else precision_value
        freeze_value = data.get("freeze_mode", FreezeMode.NONE)
        freeze_mode = FreezeMode(freeze_value) if not isinstance(freeze_value, FreezeMode) else freeze_value
        kwargs = data.get("kwargs", {}) or {}
        return cls(
            name=data["name"],
            task=data.get("task", "unknown"),
            architecture=data.get("architecture", data["name"]),
            checkpoint=data.get("checkpoint"),
            device=data.get("device", "cpu"),
            precision=precision,
            freeze_mode=freeze_mode,
            kwargs=kwargs,
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "ModelConfig":
        """Load configuration from a YAML file."""

        logger.debug("Loading model config from YAML: %s", path)
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: Path) -> "ModelConfig":
        """Load configuration from a JSON file."""

        logger.debug("Loading model config from JSON: %s", path)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)

    def with_overrides(self, **overrides: Any) -> "ModelConfig":
        """Return a copy with overrides applied."""

        data = {**self.__dict__}
        kwargs = data.get("kwargs", {}).copy()

        for key, value in overrides.items():
            if key in self.__dataclass_fields__:  # type: ignore[attr-defined]
                data[key] = value
            else:
                kwargs[key] = value

        if isinstance(data.get("precision"), str):
            data["precision"] = Precision(data["precision"])
        if isinstance(data.get("freeze_mode"), str):
            data["freeze_mode"] = FreezeMode(data["freeze_mode"])

        data["kwargs"] = kwargs
        return ModelConfig(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration to a plain dictionary."""

        return {
            "name": self.name,
            "task": self.task,
            "architecture": self.architecture,
            "checkpoint": self.checkpoint,
            "device": self.device,
            "precision": self.precision.value,
            "freeze_mode": self.freeze_mode.value,
            "kwargs": self.kwargs,
        }

    def freeze_components(self) -> Iterable[str]:
        """Return a tuple of component names that should be frozen."""

        if self.freeze_mode == FreezeMode.BACKBONE:
            return ("backbone",)
        if self.freeze_mode == FreezeMode.ALL:
            return ("all",)
        return tuple()


__all__ = [
    "ModelConfig",
    "FreezeMode",
    "Precision",
]
