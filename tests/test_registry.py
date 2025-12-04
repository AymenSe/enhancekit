import enhancekit
from enhancekit.registry import REGISTRY


def test_default_models_available():
    models = set(enhancekit.list_models())
    assert "identity" in models
    assert "identity_gain2" in models


def test_model_builds_with_freeze():
    model = enhancekit.load_model("identity", device="cpu", freeze=True)
    assert model.config.name == "identity"
    assert all(param.requires_grad is False for param in model.parameters())


def test_override_kwargs():
    model = REGISTRY.build("identity", pretrained=False, device="cpu", freeze=False, gain=3.0)
    assert float(model.gain.item()) == 3.0
