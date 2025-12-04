import torch.nn as nn

import enhancekit
from enhancekit.models import UformerModel
import enhancekit.models.uformer as uformer_module


def test_load_uformer_allows_runtime_overrides(monkeypatch):
    dummy_backbone = nn.Identity()
    monkeypatch.setattr(uformer_module, "Uformer", lambda **_: dummy_backbone)

    checkpoints = {}

    def fake_load_weights(self, checkpoint=None):
        checkpoints["checkpoint"] = checkpoint

    monkeypatch.setattr(UformerModel, "load_weights", fake_load_weights, raising=False)

    model = enhancekit.load_model("uformer", device="cpu", freeze=True, pretrained=True)

    assert isinstance(model, UformerModel)
    assert model.backbone is dummy_backbone
    assert model.device_type == "cpu"
    assert checkpoints["checkpoint"] == "weights/uformer.pth"
    assert all(param.requires_grad is False for param in model.parameters())
    assert model.config.name == "uformer"
