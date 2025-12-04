"""Public API for enhancekit."""
from __future__ import annotations

from typing import Dict, Iterable, Optional

from .config import ModelConfig
from .registry import REGISTRY


def list_models() -> Iterable[str]:
    """List all available model identifiers."""

    return REGISTRY.list()


def get_model_config(name: str) -> Optional[ModelConfig]:
    """Return the configuration for a registered model."""

    return REGISTRY.get(name)


def load_model(name: str, pretrained: bool = True, device: str = "cpu", freeze: bool = True, **overrides):
    """Load a model by name with optional overrides."""

    return REGISTRY.build(name, pretrained=pretrained, device=device, freeze=freeze, **overrides)


def describe_models() -> Dict[str, Dict[str, str]]:
    """Return a mapping of model metadata useful for documentation."""

    description = {}
    for name in list_models():
        config = get_model_config(name)
        if config:
            description[name] = {
                "task": config.task,
                "architecture": config.architecture,
                "precision": config.precision.value,
                "freeze_mode": config.freeze_mode.value,
            }
    return description


__all__ = [
    "list_models",
    "get_model_config",
    "load_model",
    "describe_models",
]
