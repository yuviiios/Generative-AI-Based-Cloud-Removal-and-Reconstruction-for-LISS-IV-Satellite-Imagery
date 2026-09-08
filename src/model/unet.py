"""
U-Net Architecture for Cloud Removal in LISS-IV Imagery.
Encoder-Decoder with skip connections.
Phase 2+: Replace/augment with Attention U-Net / Diffusion / SwinIR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ConvBlock(nn.Module):
    """Double convolution block with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, dropout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for satellite cloud removal.
    Input:  (B, in_channels, H, W)  — cloudy image
    Output: (B, out_channels, H, W) — reconstructed clear image
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_features: int = 32,
        depth: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.depth = depth
        feats = [base_features * (2 ** i) for i in range(depth)]

        # Encoder
        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for f in feats:
            self.encoders.append(EncoderBlock(in_ch, f, dropout))
            in_ch = f

        # Bottleneck
        self.bottleneck = ConvBlock(feats[-1], feats[-1] * 2, dropout)
        bn_ch = feats[-1] * 2

        # Decoder
        self.decoders = nn.ModuleList()
        for f in reversed(feats):
            self.decoders.append(DecoderBlock(bn_ch, f, f))
            bn_ch = f

        # Output head
        self.output_conv = nn.Sequential(
            nn.Conv2d(feats[0], out_channels, 1),
            nn.Sigmoid()   # Output in [0, 1] — spectral reflectance
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.output_conv(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: dict) -> UNet:
    """Build model from config dictionary."""
    model_cfg = cfg.get("model", {})
    return UNet(
        in_channels=model_cfg.get("in_channels", 3),
        out_channels=model_cfg.get("out_channels", 3),
        base_features=model_cfg.get("base_features", 32),
        depth=model_cfg.get("depth", 4),
        dropout=model_cfg.get("dropout", 0.1),
    )


if __name__ == "__main__":
    model = UNet()
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Parameters: {model.count_parameters():,}")
