"""
full_attention_unet.py — U-Net with Attention Gates + Inter-slice Attention
============================================================================
Combines both attention mechanisms:

1. Attention Gate on every skip connection (spatial focus)
2. Inter-slice Attention at bottleneck (3D context fusion)

Interface identical to UNet — drop-in replacement:
    from full_attention_unet import FullAttentionUNet as UNet
    model = UNet(in_channels=3, out_channels=1, base_channels=64)
    output = model(x)
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
        g_up = F.interpolate(g, size=x.shape[2:],
                             mode='bilinear', align_corners=True)
        att  = self.psi(F.relu(self.W_g(g_up) + self.W_x(x)))
        return x * att


class UpWithAG(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ag   = AttentionGate(F_g=in_ch, F_l=skip_ch, F_int=skip_ch // 2)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x    = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.pad(x, [0, skip.shape[3] - x.shape[3],
                           0, skip.shape[2] - x.shape[2]])
        skip = self.ag(g=x, x=skip)
        return self.conv(torch.cat([skip, x], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Inter-slice Attention
# ─────────────────────────────────────────────────────────────────────────────

class InterSliceAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        assert channels % num_heads == 0
        self.channels  = channels
        self.num_heads = num_heads
        self.head_dim  = channels // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj   = nn.Linear(channels, channels, bias=False)
        self.k_proj   = nn.Linear(channels, channels, bias=False)
        self.v_proj   = nn.Linear(channels, channels, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)
        self.norm     = nn.LayerNorm(channels)

    def forward(self, f_curr, f_prev, f_next):
        B, C, H, W = f_curr.shape

        def flatten(f):
            return f.flatten(2).permute(0, 2, 1)   # (B, H*W, C)

        curr_flat  = flatten(f_curr)
        neigh_flat = torch.cat([flatten(f_prev), flatten(f_next)], dim=1)

        Q = self.q_proj(curr_flat)
        K = self.k_proj(neigh_flat)
        V = self.v_proj(neigh_flat)

        N = H * W

        def split_heads(t, seq_len):
            return t.view(B, seq_len, self.num_heads, self.head_dim) \
                    .permute(0, 2, 1, 3)

        Q   = split_heads(Q, N)
        K   = split_heads(K, 2 * N)
        V   = split_heads(V, 2 * N)

        attn = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.scale, dim=-1)
        out  = torch.matmul(attn, V)
        out  = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        out  = self.norm(curr_flat + self.out_proj(out))

        return out.permute(0, 2, 1).view(B, C, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Full Attention U-Net
# ─────────────────────────────────────────────────────────────────────────────

class FullAttentionUNet(nn.Module):
    """
    U-Net with Attention Gates + Inter-slice Attention.

    Parameters
    ----------
    in_channels   : int  — input channels (3 for 2.5D k=1)
    out_channels  : int  — output channels (1 for binary segmentation)
    base_channels : int  — feature maps at first encoder level (default 64)
    num_heads     : int  — attention heads in inter-slice module (default 8)
    """

    def __init__(self, in_channels=3, out_channels=1,
                 base_channels=64, num_heads=8):
        super().__init__()
        b = base_channels

        # Encoder (shared — processes t-1, t, t+1)
        self.enc1       = DoubleConv(in_channels, b)
        self.enc2       = Down(b,     b * 2)
        self.enc3       = Down(b * 2, b * 4)
        self.enc4       = Down(b * 4, b * 8)
        self.bottleneck = Down(b * 8, b * 16)

        # Inter-slice attention at bottleneck
        self.inter_attn = InterSliceAttention(b * 16, num_heads=num_heads)

        # Decoder with Attention Gates
        self.dec4 = UpWithAG(b * 16, b * 8,  b * 8)
        self.dec3 = UpWithAG(b * 8,  b * 4,  b * 4)
        self.dec2 = UpWithAG(b * 4,  b * 2,  b * 2)
        self.dec1 = UpWithAG(b * 2,  b,      b)

        # Output
        self.out_conv = nn.Conv2d(b, out_channels, kernel_size=1)

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.bottleneck(e4)
        return (e1, e2, e3, e4), b

    def forward(self, x, x_prev=None, x_next=None):
        # Fallback: infer neighbours from channels
        if x_prev is None or x_next is None:
            if x.shape[1] >= 3:
                x_prev = x[:, 0:1, :, :].expand_as(x)
                x_next = x[:, -1:, :, :].expand_as(x)
            else:
                x_prev = x
                x_next = x

        # Encode all three slices
        skips_curr, b_curr = self.encode(x)
        _,          b_prev = self.encode(x_prev)
        _,          b_next = self.encode(x_next)

        # Inter-slice attention at bottleneck
        b_enhanced = self.inter_attn(b_curr, b_prev, b_next)

        # Decode with attention-gated skip connections
        e1, e2, e3, e4 = skips_curr
        d4 = self.dec4(b_enhanced, e4)
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
    model = FullAttentionUNet(in_channels=3, out_channels=1,
                              base_channels=64, num_heads=8)
    print(f'FullAttentionUNet params: {model.count_parameters():,}')

    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print(f'input={tuple(x.shape)}  output={tuple(out.shape)}')
