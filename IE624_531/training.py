"""
IE624 Quiz 4 — DDPM/DDIM Training Script
Roll: 531  |  Test classes: 0, 1, 2
"""

import os
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ─── Hyperparameters ──────────────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES  = 10
NULL_CLASS   = 10           # classifier-free guidance null token index
T            = 1000         # diffusion timesteps
BATCH_SIZE   = 128
EPOCHS       = 200
LR           = 2e-4
EMA_DECAY    = 0.9999
CFG_DROPOUT  = 0.15         # probability of using null class during training
GRAD_CLIP    = 1.0
SAVE_EVERY   = 10           # save checkpoint every N epochs
OUT_DIR      = "."          # weights.pth saved here


# ─── Cosine Noise Schedule ────────────────────────────────────────────────────

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-4, 0.9999)


class DiffusionConstants:
    def __init__(self, T=1000, device="cpu"):
        betas                = cosine_beta_schedule(T).to(device)
        alphas               = 1.0 - betas
        alphas_cumprod       = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev  = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.T                            = T
        self.betas                        = betas
        self.alphas                       = alphas
        self.alphas_cumprod               = alphas_cumprod
        self.alphas_cumprod_prev          = alphas_cumprod_prev
        self.sqrt_alphas_cumprod          = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod).sqrt()
        self.sqrt_recip_alphas            = alphas.rsqrt()
        self.posterior_variance           = (
            betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )


# ─── U-Net Building Blocks ────────────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half   = self.dim // 2
        emb    = math.log(10000) / (half - 1)
        emb    = torch.exp(torch.arange(half, device=device) * -emb)
        emb    = x[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, n_cls):
        super().__init__()
        self.norm1    = nn.GroupNorm(8, in_ch)
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2    = nn.GroupNorm(8, out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch * 2)
        self.cls_emb  = nn.Embedding(n_cls, out_ch * 2)
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, cls):
        h = self.conv1(F.silu(self.norm1(x)))

        t = self.time_mlp(F.silu(t_emb))[:, :, None, None]
        ts, tb = t.chunk(2, dim=1)

        c = self.cls_emb(cls)[:, :, None, None]
        cs, cb = c.chunk(2, dim=1)

        h = self.norm2(h) * (1 + ts + cs) + tb + cb
        h = self.conv2(F.silu(h))
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).view(B, C, H, W)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


# ─── U-Net ────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net for CIFAR-10 (32x32).
    Channel progression: 128 -> 256 -> 512
    Self-attention applied at 16x16 resolution (both encoder and decoder).
    """
    def __init__(self, base_ch=128, num_classes=NUM_CLASSES):
        super().__init__()
        n_cls    = num_classes + 1          # +1 for null CFG token
        time_dim = base_ch * 4
        ch       = [base_ch, base_ch * 2, base_ch * 4]

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.init_conv     = nn.Conv2d(3, ch[0], 3, padding=1)

        self.down1a        = ResBlock(ch[0], ch[0], time_dim, n_cls)    # 32x32
        self.down1b        = ResBlock(ch[0], ch[0], time_dim, n_cls)
        self.down1_ds      = Downsample(ch[0])                          # -> 16x16

        self.down2a        = ResBlock(ch[0], ch[1], time_dim, n_cls)    # 16x16
        self.down2b        = ResBlock(ch[1], ch[1], time_dim, n_cls)
        self.attn_down2    = SelfAttention(ch[1])
        self.down2_ds      = Downsample(ch[1])                          # -> 8x8

        self.down3a        = ResBlock(ch[1], ch[2], time_dim, n_cls)    # 8x8
        self.down3b        = ResBlock(ch[2], ch[2], time_dim, n_cls)
        self.down3_ds      = Downsample(ch[2])                          # -> 4x4

        # Bottleneck
        self.mid1          = ResBlock(ch[2], ch[2], time_dim, n_cls)
        self.mid_attn      = SelfAttention(ch[2])
        self.mid2          = ResBlock(ch[2], ch[2], time_dim, n_cls)

        # Decoder
        self.up3_us        = Upsample(ch[2])                            # 4x4 -> 8x8
        self.up3a          = ResBlock(ch[2] + ch[2], ch[2], time_dim, n_cls)
        self.up3b          = ResBlock(ch[2], ch[2], time_dim, n_cls)

        self.up2_us        = Upsample(ch[2])                            # 8x8 -> 16x16
        self.up2a          = ResBlock(ch[2] + ch[1], ch[1], time_dim, n_cls)
        self.up2b          = ResBlock(ch[1], ch[1], time_dim, n_cls)
        self.attn_up2      = SelfAttention(ch[1])

        self.up1_us        = Upsample(ch[1])                            # 16x16 -> 32x32
        self.up1a          = ResBlock(ch[1] + ch[0], ch[0], time_dim, n_cls)
        self.up1b          = ResBlock(ch[0], ch[0], time_dim, n_cls)

        self.out_norm      = nn.GroupNorm(8, ch[0])
        self.out_conv      = nn.Conv2d(ch[0], 3, 1)

    def forward(self, x, t, cls):
        t_emb = self.time_mlp(t)
        x     = self.init_conv(x)

        d1 = self.down1b(self.down1a(x, t_emb, cls), t_emb, cls)
        x1 = self.down1_ds(d1)

        d2 = self.down2a(x1, t_emb, cls)
        d2 = self.attn_down2(self.down2b(d2, t_emb, cls))
        x2 = self.down2_ds(d2)

        d3 = self.down3b(self.down3a(x2, t_emb, cls), t_emb, cls)
        x3 = self.down3_ds(d3)

        m = self.mid1(x3, t_emb, cls)
        m = self.mid_attn(m)
        m = self.mid2(m, t_emb, cls)

        u = self.up3_us(m)
        u = self.up3b(self.up3a(torch.cat([u, d3], 1), t_emb, cls), t_emb, cls)

        u = self.up2_us(u)
        u = self.up2a(torch.cat([u, d2], 1), t_emb, cls)
        u = self.attn_up2(self.up2b(u, t_emb, cls))

        u = self.up1_us(u)
        u = self.up1b(self.up1a(torch.cat([u, d1], 1), t_emb, cls), t_emb, cls)

        return self.out_conv(F.silu(self.out_norm(u)))


# ─── EMA ──────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay  = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.copy_(self.decay * s + (1.0 - self.decay) * p)

    def state_dict(self):
        return self.shadow.state_dict()


# ─── Training Loop ────────────────────────────────────────────────────────────

def train():
    print(f"Device: {DEVICE}", flush=True)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    dataset = datasets.CIFAR10(
        root="/tmp/cifar10", train=True, download=True, transform=transform
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    model   = UNet().to(DEVICE)
    ema     = EMA(model, decay=EMA_DECAY)
    optim   = torch.optim.AdamW(model.parameters(), lr=LR)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
    dc      = DiffusionConstants(T=T, device=DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}", flush=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            # CFG dropout — replace class with null token
            mask   = torch.rand(labels.size(0), device=DEVICE) < CFG_DROPOUT
            labels = torch.where(mask, torch.full_like(labels, NULL_CLASS), labels)

            # Sample random timesteps
            t = torch.randint(0, T, (imgs.size(0),), device=DEVICE).long()

            # Forward diffusion: q(x_t | x_0)
            noise  = torch.randn_like(imgs)
            sq_a   = dc.sqrt_alphas_cumprod[t][:, None, None, None]
            sq_1ma = dc.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
            x_t    = sq_a * imgs + sq_1ma * noise

            # Predict noise and compute loss
            pred  = model(x_t, t, labels)
            loss  = F.mse_loss(pred, noise)

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()
            ema.update(model)

            total_loss += loss.item()

        sched.step()
        avg = total_loss / len(loader)
        print(f"Epoch {epoch:3d}/{EPOCHS}  loss={avg:.5f}", flush=True)

        if epoch % SAVE_EVERY == 0:
            ckpt = os.path.join(OUT_DIR, f"weights_ep{epoch:03d}.pth")
            torch.save(ema.state_dict(), ckpt)
            print(f"  Checkpoint → {ckpt}", flush=True)

    # Final EMA weights
    final = os.path.join(OUT_DIR, "weights.pth")
    torch.save(ema.state_dict(), final)
    print(f"Training complete. EMA weights saved → {final}", flush=True)


if __name__ == "__main__":
    train()
