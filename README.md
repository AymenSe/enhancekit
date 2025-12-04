# enhancekit

Enhancekit provides a unified, production-ready interface around deep learning models for image enhancement and restoration. It standardizes configuration, model loading, and inference across heterogeneous model implementations.

## Features
- Central registry for available models with metadata
- Configurable precision and freezing policies
- Simple image I/O helpers with batch and folder processing
- CLI for quick experimentation

## Installation
```bash
pip install enhancekit
```

## Quickstart
```python
from enhancekit import list_models, load_model

print(list_models())
model = load_model("identity", device="cpu", freeze=True)
output_image = model.enhance_image("input.png")
output_image.save("enhanced.png")
```

## CLI usage
```bash
enhancekit --list
enhancekit identity ./examples/image.png --output ./outputs
```

## Extending
New models can be added by registering a `ModelConfig` with a constructor that subclasses `BaseEnhancementModel`. Custom freeze logic can be implemented via the `freeze` method, and new preprocessing can rely on utilities in `enhancekit.utils`.
