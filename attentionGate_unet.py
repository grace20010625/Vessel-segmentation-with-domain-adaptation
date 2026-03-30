"""
attention_unet.py — U-Net with Attention Gates on skip connections
==================================================================
Attention Gate (Oktay et al., 2018) on every skip connection.

At each decoder level, before concatenating the skip connection:
    g = gating signal from decoder (coarser, semantically rich)
    x = skip connection from encoder (finer, spatially detailed)
    attention map = sigmoid(W_psi(relu(W_g(g) + W_x(x))))
    output = x * attention_map

This suppresses irrelevant background regions and lets the decoder
focus on vessel locations, which is critical for sparse vessel data.

Interface identical to UNet — drop-in replacement:
    from attention_unet import AttentionUNet as UNet
    model = UNet(in_channels=3, out_channels=1, base_channels=64)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
# Attention Gate
# ─────────────────────────────────────────────────────────────────────────────

class AttentionGate(nn.Module):
    """
    Soft spatial attention gate on skip connections.

    F_g   : channels of gating signal g (from decoder path)
    F_l   : channels of skip connection x (from encoder path)
    F_int : intermediate channels (typically F_l // 2)
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, g, x):
        """
        g : (B, F_g, H', W')  gating signal (from decoder, may be smaller)
        x : (B, F_l, H,  W)   skip connection (from encoder)
        """
        # Upsample g to match x spatial size
        g_up = F.interpolate(g, size=x.shape[2:],
                             mode='bilinear', align_corners=True)
        att  = self.psi(F.relu(self.W_g(g_up) + self.W_x(x)))
        return x * att


# ─────────────────────────────────────────────────────────────────────────────
# Decoder block with Attention Gate
# ─────────────────────────────────────────────────────────────────────────────

class UpWithAG(nn.Module):
    """Upsample → AttentionGate on skip → concat → DoubleConv."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ag  = AttentionGate(F_g=in_ch, F_l=skip_ch, F_int=skip_ch // 2)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x    = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.pad(x, [0, skip.shape[3] - x.shape[3],
                           0, skip.shape[2] - x.shape[2]])
        skip = self.ag(g=x, x=skip)          # attended skip connection
        return self.conv(torch.cat([skip, x], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Attention U-Net
# ─────────────────────────────────────────────────────────────────────────────

class AttentionUNet(nn.Module):
    """
    U-Net with Attention Gates on all four skip connections.

    Parameters
    ----------
    in_channels   : int  — input channels (3 for 2.5D k=1, 1 for 2D)
    out_channels  : int  — output channels (1 for binary segmentation)
    base_channels : int  — feature maps at first encoder level (default 64)
    """

    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        super().__init__()
        b = base_channels

        # Encoder
        self.enc1       = DoubleConv(in_channels, b)
        self.enc2       = Down(b,     b * 2)
        self.enc3       = Down(b * 2, b * 4)
        self.enc4       = Down(b * 4, b * 8)
        self.bottleneck = Down(b * 8, b * 16)

        # Decoder with Attention Gates
        self.dec4 = UpWithAG(b * 16, b * 8,  b * 8)
        self.dec3 = UpWithAG(b * 8,  b * 4,  b * 4)
        self.dec2 = UpWithAG(b * 4,  b * 2,  b * 2)
        self.dec1 = UpWithAG(b * 2,  b,      b)

        # Output
        self.out_conv = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.bottleneck(e4)

        # Decoder with attention-gated skip connections
        d4 = self.dec4(b,  e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return self.out_conv(d1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Quick check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    for k in [0, 1, 2]:
        in_ch = 2 * k + 1
        model = AttentionUNet(in_channels=in_ch, out_channels=1, base_channels=64)
        x     = torch.randn(2, in_ch, 256, 256)
        out   = model(x)
        print(f'k={k}  in_ch={in_ch}  '
              f'input={tuple(x.shape)}  output={tuple(out.shape)}  '
              f'params={model.count_parameters():,}')
