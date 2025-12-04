"""Model registry for enhancekit."""
from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional, Type

from .config import ModelConfig
from .core import ExampleEnhancementModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central registry storing known models and configuration metadata."""

    def __init__(self) -> None:
        self._configs: Dict[str, ModelConfig] = {}
        self._constructors: Dict[str, Type] = {}

    def register(self, config: ModelConfig, constructor: Type) -> None:
        self._configs[config.name] = config
        self._constructors[config.name] = constructor
        logger.debug("Registered model %s", config.name)

    def list(self) -> Iterable[str]:
        return sorted(self._configs.keys())

    def get(self, name: str) -> Optional[ModelConfig]:
        return self._configs.get(name)

    def build(self, name: str, **overrides):
        config = self.get(name)
        if config is None:
            raise KeyError(f"Model {name} is not registered")
        runtime_keys = {"pretrained", "device", "freeze"}
        config_overrides = {k: v for k, v in overrides.items() if k not in runtime_keys}
        merged_config = config.with_overrides(**config_overrides)
        constructor = self._constructors[name]
        model = constructor(merged_config)
        if overrides.get("pretrained", True):
            model.load_weights(merged_config.checkpoint)
        freeze = overrides.get("freeze", None)
        model.freeze(freeze)
        device = overrides.get("device", merged_config.device)
        model.to(device)
        return model


def default_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register(
        ModelConfig(
            name="identity",
            task="identity",
            architecture="Identity baseline",
            checkpoint=None,
            device="cpu",
            kwargs={"gain": 1.0},
        ),
        ExampleEnhancementModel,
    )
    registry.register(
        ModelConfig(
            name="identity_gain2",
            task="identity",
            architecture="Identity gain 2",
            checkpoint=None,
            device="cpu",
            kwargs={"gain": 2.0},
        ),
        ExampleEnhancementModel,
    )
    return registry


REGISTRY = default_registry()

__all__ = ["ModelRegistry", "REGISTRY", "default_registry"]
