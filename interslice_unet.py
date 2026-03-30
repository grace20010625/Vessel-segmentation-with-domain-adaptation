"""
interslice_unet.py — U-Net with Inter-slice Attention at bottleneck
====================================================================
Standard U-Net encoder/decoder, with one addition:
    At the bottleneck, features from the adjacent slices (t-1, t+1)
    are fused into the current slice (t) via cross-attention.

    Q = current slice bottleneck features
    K = V = concatenated neighbour bottleneck features
    Output = weighted sum, added back to current slice (residual)

This is the core novelty compared to naive 2.5D (channel stacking).
Channel stacking lets the encoder passively see neighbours;
inter-slice attention lets the bottleneck actively query them.

Interface:
    model = InterSliceUNet(in_channels=3, out_channels=1, base_channels=64)

    # Standard forward (neighbours inferred from centre channel — fallback)
    logits = model(x)                        # x: (B, C, H, W)

    # Full forward with explicit neighbour volumes
    logits = model(x, x_prev, x_next)        # each: (B, C, H, W)
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


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[3] - x.shape[3],
                           0, skip.shape[2] - x.shape[2]])
        return self.conv(torch.cat([skip, x], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Inter-slice Attention Module
# ─────────────────────────────────────────────────────────────────────────────

class InterSliceAttention(nn.Module):
    """
    Cross-attention between current slice and its neighbours at bottleneck.

    Given bottleneck features of slices t-1, t, t+1 (each B x C x H x W):
      - Flatten spatial dims: B x C x (H*W) → B x (H*W) x C
      - Q  = linear(F_t)
      - K  = linear(concat[F_{t-1}, F_{t+1}], dim=sequence)
      - V  = linear(concat[F_{t-1}, F_{t+1}], dim=sequence)
      - Attention = softmax(Q K^T / sqrt(d)) V
      - Output    = F_t + project(Attention)   [residual]

    Parameters
    ----------
    channels  : int  — bottleneck feature channels (b * 16)
    num_heads : int  — number of attention heads (default 8)
    """

    def __init__(self, channels, num_heads=8):
        super().__init__()
        assert channels % num_heads == 0, \
            f'channels ({channels}) must be divisible by num_heads ({num_heads})'

        self.channels  = channels
        self.num_heads = num_heads
        self.head_dim  = channels // num_heads
        self.scale     = self.head_dim ** -0.5

        # Projections for current slice (query)
        self.q_proj = nn.Linear(channels, channels, bias=False)
        # Projections for neighbours (key, value)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        # Output projection
        self.out_proj = nn.Linear(channels, channels, bias=False)
        # Layer norm + residual
        self.norm = nn.LayerNorm(channels)

    def forward(self, f_curr, f_prev, f_next):
        """
        f_curr : (B, C, H, W)  — current slice bottleneck
        f_prev : (B, C, H, W)  — previous slice bottleneck
        f_next : (B, C, H, W)  — next slice bottleneck

        Returns (B, C, H, W) — enhanced current slice features
        """
        B, C, H, W = f_curr.shape

        # Flatten spatial: (B, C, H, W) → (B, H*W, C)
        def flatten(f):
            return f.flatten(2).permute(0, 2, 1)   # (B, H*W, C)

        curr_flat = flatten(f_curr)                 # (B, N, C)  N = H*W
        prev_flat = flatten(f_prev)                 # (B, N, C)
        next_flat = flatten(f_next)                 # (B, N, C)

        # Neighbours as key/value sequence: (B, 2N, C)
        neigh_flat = torch.cat([prev_flat, next_flat], dim=1)

        # Project
        Q = self.q_proj(curr_flat)                  # (B, N, C)
        K = self.k_proj(neigh_flat)                 # (B, 2N, C)
        V = self.v_proj(neigh_flat)                 # (B, 2N, C)

        # Split heads
        def split_heads(t, seq_len):
            return t.view(B, seq_len, self.num_heads, self.head_dim) \
                    .permute(0, 2, 1, 3)            # (B, heads, seq, head_dim)

        N  = H * W
        Q  = split_heads(Q, N)                      # (B, heads, N, head_dim)
        K  = split_heads(K, 2 * N)                  # (B, heads, 2N, head_dim)
        V  = split_heads(V, 2 * N)                  # (B, heads, 2N, head_dim)

        # Scaled dot-product attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale   # (B, heads, N, 2N)
        attn = F.softmax(attn, dim=-1)

        out  = torch.matmul(attn, V)                # (B, heads, N, head_dim)
        out  = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)  # (B, N, C)

        # Output projection + residual + norm
        out  = self.out_proj(out)
        out  = self.norm(curr_flat + out)           # (B, N, C)

        # Reshape back to spatial: (B, N, C) → (B, C, H, W)
        out  = out.permute(0, 2, 1).view(B, C, H, W)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Inter-slice U-Net
# ─────────────────────────────────────────────────────────────────────────────

class InterSliceUNet(nn.Module):
    """
    U-Net with Inter-slice Attention at the bottleneck.

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

        # Encoder (shared weights — processes t-1, t, t+1 independently)
        self.enc1      = DoubleConv(in_channels, b)
        self.enc2      = Down(b,     b * 2)
        self.enc3      = Down(b * 2, b * 4)
        self.enc4      = Down(b * 4, b * 8)
        self.bottleneck= Down(b * 8, b * 16)

        # Inter-slice attention at bottleneck
        self.inter_attn = InterSliceAttention(b * 16, num_heads=num_heads)

        # Decoder
        self.dec4 = Up(b * 16, b * 8,  b * 8)
        self.dec3 = Up(b * 8,  b * 4,  b * 4)
        self.dec2 = Up(b * 4,  b * 2,  b * 2)
        self.dec1 = Up(b * 2,  b,      b)

        # Output
        self.out_conv = nn.Conv2d(b, out_channels, kernel_size=1)

    def encode(self, x):
        """Run encoder and return (skip connections, bottleneck features)."""
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.bottleneck(e4)
        return (e1, e2, e3, e4), b

    def decode(self, b, skips):
        """Run decoder given bottleneck and skip connections."""
        e1, e2, e3, e4 = skips
        d4 = self.dec4(b,  e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.out_conv(d1)

    def forward(self, x, x_prev=None, x_next=None):
        """
        x      : (B, C, H, W) — current slice input (centre channel = t)
        x_prev : (B, C, H, W) — previous slice input (optional)
        x_next : (B, C, H, W) — next slice input (optional)

        If x_prev / x_next are None, they are inferred from x's channels:
            x_prev = x with only channel 0 (replicated to C channels)
            x_next = x with only channel 2 (replicated to C channels)
        This fallback keeps the interface compatible with the base DataLoader.
        """
        # Fallback: extract neighbours from multi-channel input
        if x_prev is None or x_next is None:
            C = x.shape[1]
            if C >= 3:
                # 2.5D input: [t-1, t, t+1] stacked as channels
                # Use each channel repeated C times as a proxy volume
                x_prev = x[:, 0:1, :, :].expand_as(x)
                x_next = x[:, -1:, :, :].expand_as(x)
            else:
                # Pure 2D: no neighbour info, use x itself
                x_prev = x
                x_next = x

        # Encode all three slices (shared encoder weights)
        skips_curr, b_curr = self.encode(x)
        _,          b_prev = self.encode(x_prev)
        _,          b_next = self.encode(x_next)

        # Inter-slice attention at bottleneck
        b_enhanced = self.inter_attn(b_curr, b_prev, b_next)

        # Decode using current slice skips + enhanced bottleneck
        return self.decode(b_enhanced, skips_curr)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Quick check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    B, C, H, W = 2, 3, 256, 256

    model = InterSliceUNet(in_channels=C, out_channels=1,
                           base_channels=64, num_heads=8)
    print(f'InterSliceUNet params: {model.count_parameters():,}')

    x = torch.randn(B, C, H, W)

    # Test standard forward (neighbours inferred)
    out = model(x)
    print(f'Standard forward:  input={tuple(x.shape)}  output={tuple(out.shape)}')

    # Test full forward with explicit neighbours
    x_prev = torch.randn(B, C, H, W)
    x_next = torch.randn(B, C, H, W)
    out2 = model(x, x_prev, x_next)
    print(f'Full forward:      input={tuple(x.shape)}  output={tuple(out2.shape)}')
