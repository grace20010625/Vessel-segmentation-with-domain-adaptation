"""
unet.py — Standard U-Net with support for multi-channel 2.5D input
====================================================================
Architecture follows Ronneberger et al. (2015) with minor modern fixes:
  - BatchNorm after each conv (original used no BN)
  - Padding=1 to preserve spatial size (original had no padding → crop)
  - in_channels configurable for 2.5D input (1, 3, 5, ...)

Usage:
    from unet import UNet
    model = UNet(in_channels=3, out_channels=1, base_channels=64)
    output = model(input)   # input: (B, C, H, W) → output: (B, 1, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Conv → BN → ReLU → Conv → BN → ReLU"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool2d → DoubleConv"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """
    Bilinear upsample → concatenate skip connection → DoubleConv.
    Uses bilinear upsampling (no transposed conv) for stability.
    """

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if spatial size mismatch (can happen with odd input sizes)
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[3] - x.shape[3],
                           0, skip.shape[2] - x.shape[2]])
        return self.conv(torch.cat([skip, x], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# U-Net
# ─────────────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Standard U-Net (4 levels of downsampling).

    Parameters
    ----------
    in_channels   : int  — input channels (1 for 2D, 3 for 2.5D k=1, 5 for k=2)
    out_channels  : int  — output channels (1 for binary segmentation)
    base_channels : int  — feature maps at first encoder level (default 64)
    """

    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        super().__init__()
        b = base_channels

        # Encoder
        self.enc1 = DoubleConv(in_channels, b)        # → (B, b,   H,   W)
        self.enc2 = Down(b,     b * 2)                # → (B, 2b,  H/2, W/2)
        self.enc3 = Down(b * 2, b * 4)                # → (B, 4b,  H/4, W/4)
        self.enc4 = Down(b * 4, b * 8)                # → (B, 8b,  H/8, W/8)

        # Bottleneck
        self.bottleneck = Down(b * 8, b * 16)         # → (B, 16b, H/16, W/16)

        # Decoder
        self.dec4 = Up(b * 16, b * 8,  b * 8)        # → (B, 8b,  H/8, W/8)
        self.dec3 = Up(b * 8,  b * 4,  b * 4)        # → (B, 4b,  H/4, W/4)
        self.dec2 = Up(b * 4,  b * 2,  b * 2)        # → (B, 2b,  H/2, W/2)
        self.dec1 = Up(b * 2,  b,      b)             # → (B, b,   H,   W)

        # Output
        self.out_conv = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b  = self.bottleneck(e4)

        # Decoder with skip connections
        d4 = self.dec4(b,  e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return self.out_conv(d1)   # raw logits (B, 1, H, W)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Quick check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    for k in [0, 1, 2]:
        in_ch = 2 * k + 1
        model = UNet(in_channels=in_ch, out_channels=1, base_channels=64)
        x     = torch.randn(2, in_ch, 512, 512)
        y     = model(x)
        print(f'k={k}  in_channels={in_ch}  '
              f'input={tuple(x.shape)}  output={tuple(y.shape)}  '
              f'params={model.count_parameters():,}')
