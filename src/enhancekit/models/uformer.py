"""Wrappers for transformer-based image restoration models."""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..core import BaseEnhancementModel


class LayerNorm2d(nn.Module):
    """Channel-wise layer normalization for 2D feature maps."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x + self.bias


class MultiDconvHeadAttention(nn.Module):
    """Multi-head attention with depthwise convolution as in Restormer."""

    def __init__(self, dim: int, num_heads: int = 4, bias: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        def reshape_tensor(tensor: Tensor) -> Tensor:
            return tensor.reshape(b, self.num_heads, c // self.num_heads, h * w)

        q = reshape_tensor(q)
        k = reshape_tensor(k)
        v = reshape_tensor(v)
        q = F.normalize(q, dim=2)
        k = F.normalize(k, dim=2)

        attn = torch.matmul(q.transpose(2, 3), k) * self.temperature
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v.transpose(2, 3))
        out = out.transpose(2, 3).reshape(b, c, h, w)
        out = self.project_out(out)
        return out


class GatedDconvFeedForward(nn.Module):
    """Gated depthwise convolutional feed-forward network."""

    def __init__(self, dim: int, expansion_factor: float = 2.66, bias: bool = True) -> None:
        super().__init__()
        hidden_features = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    """Restormer transformer block with attention and feed-forward network."""

    def __init__(
        self, dim: int, num_heads: int, ffn_expansion_factor: float = 2.66, bias: bool = True
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MultiDconvHeadAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GatedDconvFeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.body(x)


class Restormer(nn.Module):
    """Restormer encoder-decoder network."""

    def __init__(
        self,
        inp_channels: int = 3,
        embed_dim: int = 48,
        depths: tuple[int, int, int, int] | None = None,
        num_heads: tuple[int, int, int, int] | None = None,
        ffn_expansion_factor: float = 2.66,
        bias: bool = True,
    ) -> None:
        super().__init__()
        depths = depths or (4, 6, 6, 8)
        num_heads = num_heads or (1, 2, 4, 8)
        dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]

        self.patch_embed = nn.Conv2d(inp_channels, embed_dim, kernel_size=3, stride=1, padding=1)

        # Encoder
        self.encoder_levels = nn.ModuleList()
        for idx, depth in enumerate(depths):
            blocks = nn.Sequential(
                *[TransformerBlock(dims[idx], num_heads[idx], ffn_expansion_factor, bias) for _ in range(depth)]
            )
            self.encoder_levels.append(blocks)

        self.downsamples = nn.ModuleList(
            [Downsample(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            *[TransformerBlock(dims[-1], num_heads[-1], ffn_expansion_factor, bias) for _ in range(4)]
        )

        # Decoder
        self.upsamples = nn.ModuleList(
            [Upsample(dims[i + 1], dims[i]) for i in reversed(range(len(dims) - 1))]
        )
        self.decoder_levels = nn.ModuleList()
        for i, depth in enumerate(reversed(depths)):
            dim = dims[len(dims) - i - 2]
            self.decoder_levels.append(
                nn.Sequential(
                    *[TransformerBlock(dim * 2 if j == 0 else dim, num_heads[len(dims) - i - 2], ffn_expansion_factor, bias) for j in range(depth)]
                )
            )

        self.output = nn.Conv2d(embed_dim, inp_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = self.patch_embed(x)
        skips: list[Tensor] = []
        for encoder, down in zip(self.encoder_levels, self.downsamples):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for decoder, up, skip in zip(self.decoder_levels, self.upsamples, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        return self.output(x)


class SpatialTransformerBlock(nn.Module):
    """Simplified transformer block for Uformer-style processing."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # B, HW, C
        tokens_norm = self.norm1(tokens)
        attn_out, _ = self.attn(tokens_norm, tokens_norm, tokens_norm)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class UformerBackbone(nn.Module):
    """Lightweight Uformer-style encoder-decoder with transformer blocks."""

    def __init__(
        self,
        inp_channels: int = 3,
        embed_dim: int = 32,
        depths: tuple[int, int, int] | None = None,
        num_heads: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        depths = depths or (2, 2, 4)
        num_heads = num_heads or (2, 4, 8)
        dims = [embed_dim, embed_dim * 2, embed_dim * 4]

        self.stem = nn.Conv2d(inp_channels, embed_dim, kernel_size=3, stride=1, padding=1)

        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i, depth in enumerate(depths):
            dim = dims[i]
            blocks = nn.Sequential(
                *[SpatialTransformerBlock(dim, num_heads=min(num_heads[i], dim), mlp_ratio=2.0) for _ in range(depth)]
            )
            self.encoders.append(blocks)
            if i < len(depths) - 1:
                self.downsamples.append(nn.Conv2d(dim, dims[i + 1], kernel_size=3, stride=2, padding=1))

        self.bottleneck = nn.Sequential(
            *[SpatialTransformerBlock(dims[-1], num_heads[-1], mlp_ratio=2.0) for _ in range(depths[-1])]
        )

        self.upsamples = nn.ModuleList(
            [nn.ConvTranspose2d(dims[i + 1], dims[i], kernel_size=2, stride=2) for i in reversed(range(len(dims) - 1))]
        )
        self.decoders = nn.ModuleList()
        for i, depth in enumerate(reversed(depths[:-1])):
            dim = dims[len(dims) - i - 2]
            self.decoders.append(
                nn.Sequential(
                    *[SpatialTransformerBlock(dim * 2 if j == 0 else dim, num_heads=min(num_heads[len(dims) - i - 2], dim), mlp_ratio=2.0) for j in range(depth)]
                )
            )

        self.output = nn.Conv2d(embed_dim, inp_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = self.stem(x)
        skips: list[Tensor] = []
        for encoder, down in zip(self.encoders, self.downsamples):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for decoder, up, skip in zip(self.decoders, self.upsamples, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        return self.output(x)


class RestormerModel(BaseEnhancementModel):
    """Wrapper around the Restormer architecture."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        kwargs = config.kwargs or {}
        self.backbone = Restormer(
            inp_channels=kwargs.get("inp_channels", 3),
            embed_dim=kwargs.get("embed_dim", 48),
            depths=tuple(kwargs.get("depths", (4, 6, 6, 8))),
            num_heads=tuple(kwargs.get("num_heads", (1, 2, 4, 8))),
            ffn_expansion_factor=kwargs.get("ffn_expansion_factor", 2.66),
            bias=kwargs.get("bias", True),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.backbone(x)


class UformerModel(BaseEnhancementModel):
    """Wrapper around a compact Uformer-style network."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        kwargs = config.kwargs or {}
        self.backbone = UformerBackbone(
            inp_channels=kwargs.get("inp_channels", 3),
            embed_dim=kwargs.get("embed_dim", 32),
            depths=tuple(kwargs.get("depths", (2, 2, 4))),
            num_heads=tuple(kwargs.get("num_heads", (2, 4, 8))),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.backbone(x)


__all__ = [
    "Restormer",
    "RestormerModel",
    "UformerBackbone",
    "UformerModel",
    "TransformerBlock",
    "SpatialTransformerBlock",
]
