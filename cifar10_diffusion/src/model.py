"""
Conditional U-Net for DDPM / DDIM on CIFAR-10 (32×32).

Architecture  (Ho et al. 2020, improved by Dhariwal & Nichol 2021):
  32×32 (ch=128)  →  16×16 (ch=256, attn)  →  8×8 (ch=256, attn)
        ↕ middle block at 8×8 (ch=256, attn)
  8×8 → 16×16 → 32×32

Conditioning:
  - Timestep  : sinusoidal → MLP → emb_dim
  - Class label: nn.Embedding (class 10 = null/unconditional for CFG)
  Both are added and injected into every ResBlock via Adaptive Group Norm (AdaGN).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def get_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal position encoding for diffusion timesteps."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


# ─────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    ResNet block with AdaGN conditioning (scale + shift from embedding).
    """

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        # Projects embedding to (scale, shift) for AdaGN
        self.emb_proj = nn.Linear(emb_dim, out_ch * 2)

        self.norm2   = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2   = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        scale_shift = self.emb_proj(F.silu(emb))[:, :, None, None]  # (B, 2*out_ch, 1, 1)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale) + shift

        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class SelfAttentionBlock(nn.Module):
    """Multi-head self-attention (spatial)."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        assert channels % num_heads == 0
        self.norm     = nn.GroupNorm(32, channels)
        self.num_heads = num_heads
        self.head_dim  = channels // num_heads
        self.to_qkv   = nn.Conv1d(channels, channels * 3, 1, bias=False)
        self.to_out   = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W)

        qkv = self.to_qkv(h)                                    # (B, 3C, HW)
        q, k, v = qkv.chunk(3, dim=1)

        def split_heads(t):
            return t.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)  # (B, heads, HW, head_dim)
        out = F.scaled_dot_product_attention(q, k, v)             # (B, heads, HW, head_dim)

        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H * W)
        out = self.to_out(out).view(B, C, H, W)
        return x + out


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode='nearest'))


# ─────────────────────────────────────────────────────────────
# U-Net
# ─────────────────────────────────────────────────────────────

class ConditionalUNet(nn.Module):
    """
    Class-conditional U-Net noise predictor for CIFAR-10.

    Args:
        in_channels : image channels (3 for RGB)
        base_ch     : base feature channels (128 by default)
        num_classes : 10 for CIFAR-10; internally uses num_classes+1 embeddings
                      so that index `num_classes` serves as the null/unconditional token
                      for classifier-free guidance.
        dropout     : dropout probability in ResBlocks
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_ch:     int = 128,
        num_classes: int = 10,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.base_ch     = base_ch
        self.num_classes = num_classes
        ch   = base_ch
        ch2  = base_ch * 2
        edim = base_ch * 4      # embedding dimension

        # ── Embeddings ────────────────────────────────────────────
        self.time_embed = nn.Sequential(
            nn.Linear(ch, edim),
            nn.SiLU(),
            nn.Linear(edim, edim),
        )
        # class 0-9 = real classes, class 10 = null (unconditional)
        self.class_embed = nn.Embedding(num_classes + 1, edim)

        # ── Encoder ───────────────────────────────────────────────
        self.init_conv = nn.Conv2d(in_channels, ch, 3, padding=1)

        # Level 1  –  32×32, ch=128, no attention
        self.enc1_1 = ResBlock(ch,  ch,  edim, dropout)
        self.enc1_2 = ResBlock(ch,  ch,  edim, dropout)
        self.down1  = Downsample(ch)                          # → 16×16

        # Level 2  –  16×16, ch=256, with attention
        self.enc2_1  = ResBlock(ch,  ch2, edim, dropout)
        self.attn2_1 = SelfAttentionBlock(ch2)
        self.enc2_2  = ResBlock(ch2, ch2, edim, dropout)
        self.attn2_2 = SelfAttentionBlock(ch2)
        self.down2   = Downsample(ch2)                        # → 8×8

        # Level 3  –  8×8, ch=256, with attention
        self.enc3_1  = ResBlock(ch2, ch2, edim, dropout)
        self.attn3_1 = SelfAttentionBlock(ch2)
        self.enc3_2  = ResBlock(ch2, ch2, edim, dropout)
        self.attn3_2 = SelfAttentionBlock(ch2)

        # ── Middle ────────────────────────────────────────────────
        self.mid1     = ResBlock(ch2, ch2, edim, dropout)
        self.mid_attn = SelfAttentionBlock(ch2)
        self.mid2     = ResBlock(ch2, ch2, edim, dropout)

        # ── Decoder ───────────────────────────────────────────────
        # Level 3  –  8×8, ch=256
        #   skip from enc3_2 (ch2) and enc3_1 (ch2)
        self.dec3_1   = ResBlock(ch2 + ch2, ch2, edim, dropout)   # 512 → 256
        self.dattn3_1 = SelfAttentionBlock(ch2)
        self.dec3_2   = ResBlock(ch2 + ch2, ch2, edim, dropout)   # 512 → 256
        self.dattn3_2 = SelfAttentionBlock(ch2)
        self.up2      = Upsample(ch2)                              # → 16×16

        # Level 2  –  16×16, ch=256
        #   skip from enc2_2 (ch2) and enc2_1 (ch2)
        self.dec2_1   = ResBlock(ch2 + ch2, ch2, edim, dropout)   # 512 → 256
        self.dattn2_1 = SelfAttentionBlock(ch2)
        self.dec2_2   = ResBlock(ch2 + ch2, ch2, edim, dropout)   # 512 → 256
        self.dattn2_2 = SelfAttentionBlock(ch2)
        self.up1      = Upsample(ch2)                              # → 32×32 (still ch2)

        # Level 1  –  32×32
        #   skip from enc1_2 (ch) and enc1_1 (ch)
        self.dec1_1 = ResBlock(ch2 + ch, ch, edim, dropout)       # 384 → 128
        self.dec1_2 = ResBlock(ch  + ch, ch, edim, dropout)       # 256 → 128

        # ── Output ────────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)

        # Zero-initialise the output projection (standard for diffusion)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    # ── Forward ───────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,   # (B, 3, 32, 32)  noisy image
        t: torch.Tensor,   # (B,)             timestep indices [0, T-1]
        c: torch.Tensor,   # (B,)             class labels [0, 9]  or  num_classes (null)
    ) -> torch.Tensor:

        # Conditioning
        t_emb = get_timestep_embedding(t, self.base_ch)        # (B, base_ch)
        emb   = self.time_embed(t_emb) + self.class_embed(c)   # (B, edim)

        # Init conv
        h = self.init_conv(x)                                   # (B, ch, 32, 32)

        # ── Encoder ────────────────────────────────────────────
        h1  = self.enc1_1(h,  emb)                             # (B, ch,  32, 32)
        h2  = self.enc1_2(h1, emb)                             # (B, ch,  32, 32)
        hd1 = self.down1(h2)                                   # (B, ch,  16, 16)

        h3  = self.attn2_1(self.enc2_1(hd1, emb))             # (B, ch2, 16, 16)
        h4  = self.attn2_2(self.enc2_2(h3,  emb))             # (B, ch2, 16, 16)
        hd2 = self.down2(h4)                                   # (B, ch2,  8,  8)

        h5  = self.attn3_1(self.enc3_1(hd2, emb))             # (B, ch2,  8,  8)
        h6  = self.attn3_2(self.enc3_2(h5,  emb))             # (B, ch2,  8,  8)

        # ── Middle ─────────────────────────────────────────────
        m = self.mid1(h6, emb)                                 # (B, ch2,  8,  8)
        m = self.mid_attn(m)
        m = self.mid2(m, emb)                                  # (B, ch2,  8,  8)

        # ── Decoder ────────────────────────────────────────────
        d = self.dattn3_1(self.dec3_1(torch.cat([m,  h6], 1), emb))   # (B, ch2, 8, 8)
        d = self.dattn3_2(self.dec3_2(torch.cat([d,  h5], 1), emb))   # (B, ch2, 8, 8)
        d = self.up2(d)                                                 # (B, ch2, 16, 16)

        d = self.dattn2_1(self.dec2_1(torch.cat([d, h4], 1), emb))    # (B, ch2, 16, 16)
        d = self.dattn2_2(self.dec2_2(torch.cat([d, h3], 1), emb))    # (B, ch2, 16, 16)
        d = self.up1(d)                                                 # (B, ch2, 32, 32)

        d = self.dec1_1(torch.cat([d, h2], 1), emb)                    # (B, ch, 32, 32)
        d = self.dec1_2(torch.cat([d, h1], 1), emb)                    # (B, ch, 32, 32)

        return self.out_conv(F.silu(self.out_norm(d)))                  # (B, 3, 32, 32)


# ─────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = ConditionalUNet().to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.2f} M")

    x = torch.randn(4, 3, 32, 32, device=device)
    t = torch.randint(0, 1000, (4,), device=device)
    c = torch.randint(0, 10,   (4,), device=device)
    out = model(x, t, c)
    print(f"Input: {x.shape}  →  Output: {out.shape}")
