"""Model registry for enhancekit."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Type

from .config import ModelConfig
from .core import ExampleEnhancementModel
from .models import RestormerModel, UformerModel

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

    def register_many(self, configs: Sequence[ModelConfig], constructor: Type) -> None:
        for config in configs:
            self.register(config, constructor)

    def register_from_json(self, path: Path, constructor: Type) -> None:
        if not path.exists():
            logger.warning("Model configuration file %s not found; skipping", path)
            return
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        configs = [ModelConfig.from_dict(item) for item in data]
        self.register_many(configs, constructor)

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
    model_sources = Path(__file__).resolve().parent / "model_sources.json"
    if model_sources.exists():
        with model_sources.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        for item in data:
            config = ModelConfig.from_dict(item)
            architecture = config.architecture.lower()
            if "restormer" in architecture:
                constructor = RestormerModel
            elif "uformer" in architecture:
                constructor = UformerModel
            else:
                constructor = ExampleEnhancementModel
            registry.register(config, constructor)
    else:
        logger.warning("Model configuration file %s not found; skipping", model_sources)
    return registry


REGISTRY = default_registry()

__all__ = ["ModelRegistry", "REGISTRY", "default_registry"]
