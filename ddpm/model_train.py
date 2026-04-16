"""
model_train.py — Self-contained CIFAR-10 DDPM

Contains:
  - ConditionalUNet  : class-conditional U-Net noise predictor
  - GaussianDiffusion: forward / reverse process (DDPM + DDIM + CFG)
  - EMA              : exponential moving average of weights
  - train()          : full training loop

Usage (local):
    python model_train.py
    python model_train.py --epochs 1000 --base_ch 128
    python model_train.py --resume weights.pth
"""

import os
import copy
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Timestep embedding
# ─────────────────────────────────────────────────────────────────────────────

def get_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half  = dim // 2
    freqs = torch.exp(
        -math.log(10_000) *
        torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# U-Net building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1    = nn.GroupNorm(32, in_ch)
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch * 2)
        self.norm2    = nn.GroupNorm(32, out_ch)
        self.dropout  = nn.Dropout(dropout)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip     = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb_proj(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.norm      = nn.GroupNorm(32, channels)
        self.num_heads = num_heads
        self.head_dim  = channels // num_heads
        self.to_qkv    = nn.Conv1d(channels, channels * 3, 1, bias=False)
        self.to_out    = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h   = self.norm(x).view(B, C, H * W)
        qkv = self.to_qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        def split_heads(t):
            return t.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        out = F.scaled_dot_product_attention(split_heads(q), split_heads(k), split_heads(v))
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H * W)
        return x + self.to_out(out).view(B, C, H, W)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode='nearest'))


# ─────────────────────────────────────────────────────────────────────────────
# Conditional U-Net
# ─────────────────────────────────────────────────────────────────────────────

class ConditionalUNet(nn.Module):
    """
    Class-conditional U-Net for CIFAR-10 (32×32).
    Architecture: 32×32 (ch) → 16×16 (2ch, attn) → 8×8 (2ch, attn) → middle → decoder
    Conditioning: sinusoidal timestep + class embedding injected via AdaGN.
    """

    def __init__(self, in_channels=3, base_ch=128, num_classes=10, dropout=0.1):
        super().__init__()
        self.base_ch = base_ch
        ch   = base_ch
        ch2  = base_ch * 2
        edim = base_ch * 4

        self.time_embed  = nn.Sequential(nn.Linear(ch, edim), nn.SiLU(), nn.Linear(edim, edim))
        self.class_embed = nn.Embedding(num_classes + 1, edim)   # index num_classes = null token

        self.init_conv = nn.Conv2d(in_channels, ch, 3, padding=1)

        # Encoder
        self.enc1_1, self.enc1_2 = ResBlock(ch,  ch,  edim, dropout), ResBlock(ch,  ch,  edim, dropout)
        self.down1 = Downsample(ch)

        self.enc2_1,  self.attn2_1 = ResBlock(ch,  ch2, edim, dropout), SelfAttentionBlock(ch2)
        self.enc2_2,  self.attn2_2 = ResBlock(ch2, ch2, edim, dropout), SelfAttentionBlock(ch2)
        self.down2 = Downsample(ch2)

        self.enc3_1,  self.attn3_1 = ResBlock(ch2, ch2, edim, dropout), SelfAttentionBlock(ch2)
        self.enc3_2,  self.attn3_2 = ResBlock(ch2, ch2, edim, dropout), SelfAttentionBlock(ch2)

        # Middle
        self.mid1, self.mid_attn, self.mid2 = (
            ResBlock(ch2, ch2, edim, dropout),
            SelfAttentionBlock(ch2),
            ResBlock(ch2, ch2, edim, dropout),
        )

        # Decoder
        self.dec3_1,  self.dattn3_1 = ResBlock(ch2 + ch2, ch2, edim, dropout), SelfAttentionBlock(ch2)
        self.dec3_2,  self.dattn3_2 = ResBlock(ch2 + ch2, ch2, edim, dropout), SelfAttentionBlock(ch2)
        self.up2 = Upsample(ch2)

        self.dec2_1,  self.dattn2_1 = ResBlock(ch2 + ch2, ch2, edim, dropout), SelfAttentionBlock(ch2)
        self.dec2_2,  self.dattn2_2 = ResBlock(ch2 + ch2, ch2, edim, dropout), SelfAttentionBlock(ch2)
        self.up1 = Upsample(ch2)

        self.dec1_1 = ResBlock(ch2 + ch, ch,  edim, dropout)
        self.dec1_2 = ResBlock(ch  + ch, ch,  edim, dropout)

        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, t, c):
        emb = self.time_embed(get_timestep_embedding(t, self.base_ch)) + self.class_embed(c)

        h   = self.init_conv(x)
        h1  = self.enc1_1(h,  emb);  h2  = self.enc1_2(h1, emb)
        hd1 = self.down1(h2)

        h3  = self.attn2_1(self.enc2_1(hd1, emb));  h4  = self.attn2_2(self.enc2_2(h3,  emb))
        hd2 = self.down2(h4)

        h5  = self.attn3_1(self.enc3_1(hd2, emb));  h6  = self.attn3_2(self.enc3_2(h5,  emb))

        m = self.mid1(h6, emb)
        m = self.mid_attn(m)
        m = self.mid2(m, emb)

        d = self.dattn3_1(self.dec3_1(torch.cat([m,  h6], 1), emb))
        d = self.dattn3_2(self.dec3_2(torch.cat([d,  h5], 1), emb))
        d = self.up2(d)

        d = self.dattn2_1(self.dec2_1(torch.cat([d, h4], 1), emb))
        d = self.dattn2_2(self.dec2_2(torch.cat([d, h3], 1), emb))
        d = self.up1(d)

        d = self.dec1_1(torch.cat([d, h2], 1), emb)
        d = self.dec1_2(torch.cat([d, h1], 1), emb)
        return self.out_conv(F.silu(self.out_norm(d)))


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian Diffusion  (DDPM + DDIM + CFG)
# ─────────────────────────────────────────────────────────────────────────────

class GaussianDiffusion:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.T      = T
        self.device = device

        betas               = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
        alphas              = 1.0 - betas
        alphas_cumprod      = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]])

        r = lambda x: x.float().to(device)

        self.betas                          = r(betas)
        self.alphas_cumprod                 = r(alphas_cumprod)
        self.alphas_cumprod_prev            = r(alphas_cumprod_prev)
        self.sqrt_alphas_cumprod            = r(alphas_cumprod.sqrt())
        self.sqrt_one_minus_alphas_cumprod  = r((1.0 - alphas_cumprod).sqrt())
        self.posterior_variance             = r(betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.posterior_mean_coef1           = r(betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod))
        self.posterior_mean_coef2           = r((1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod))

    # ── Training loss ────────────────────────────────────────────────────────

    def p_losses(self, model, x0, t, c, p_uncond=0.1, null_class=10):
        noise    = torch.randn_like(x0)
        s1       = self.sqrt_alphas_cumprod[t][:, None, None, None]
        s2       = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        x_noisy  = s1 * x0 + s2 * noise

        c_train          = c.clone()
        c_train[torch.rand(c.shape[0], device=c.device) < p_uncond] = null_class

        return F.mse_loss(model(x_noisy, t, c_train), noise)

    # ── CFG noise prediction ─────────────────────────────────────────────────

    def _predict(self, model, x_t, t_batch, c, guidance_scale=1.0, null_class=10):
        if guidance_scale == 1.0:
            return model(x_t, t_batch, c)
        B      = x_t.shape[0]
        c_null = torch.full_like(c, null_class)
        eps    = model(torch.cat([x_t, x_t]), torch.cat([t_batch, t_batch]), torch.cat([c, c_null]))
        eps_c, eps_u = eps[:B], eps[B:]
        return eps_u + guidance_scale * (eps_c - eps_u)

    # ── DDPM sampling ────────────────────────────────────────────────────────

    @torch.no_grad()
    def ddpm_sample(self, model, shape, label_class, device, guidance_scale=3.0, null_class=10):
        B = shape[0]
        x = torch.randn(shape, device=device)
        c = torch.full((B,), label_class, dtype=torch.long, device=device)

        for t_idx in reversed(range(self.T)):
            t_b  = torch.full((B,), t_idx, dtype=torch.long, device=device)
            eps  = self._predict(model, x, t_b, c, guidance_scale, null_class)
            x0p  = ((x - self.sqrt_one_minus_alphas_cumprod[t_idx] * eps)
                    / self.sqrt_alphas_cumprod[t_idx]).clamp(-1, 1)
            mean = self.posterior_mean_coef1[t_idx] * x0p + self.posterior_mean_coef2[t_idx] * x
            x    = mean if t_idx == 0 else mean + self.posterior_variance[t_idx].sqrt() * torch.randn_like(x)

        return (x.clamp(-1, 1) + 1.0) / 2.0

    # ── DDIM sampling ────────────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(self, model, shape, label_class, device,
                    steps=50, eta=0.0, guidance_scale=3.0, null_class=10):
        B         = shape[0]
        x         = torch.randn(shape, device=device)
        c         = torch.full((B,), label_class, dtype=torch.long, device=device)
        step_size = self.T // steps
        timesteps = list(reversed(range(0, self.T, step_size)))[:steps]

        for i, t_idx in enumerate(timesteps):
            t_b   = torch.full((B,), t_idx, dtype=torch.long, device=device)
            eps   = self._predict(model, x, t_b, c, guidance_scale, null_class)
            ab_t  = self.alphas_cumprod[t_idx]
            x0p   = ((x - (1.0 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-1, 1)

            if i == len(timesteps) - 1:
                x = x0p
            else:
                ab_prev   = self.alphas_cumprod[timesteps[i + 1]]
                sigma     = eta * ((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)).clamp(min=0).sqrt()
                direction = (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
                x         = ab_prev.sqrt() * x0p + direction + (sigma * torch.randn_like(x) if eta > 0 else 0.0)

        return (x.clamp(-1, 1) + 1.0) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay  = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        raw = model.module if isinstance(model, nn.DataParallel) else model
        for s, m in zip(self.shadow.parameters(), raw.parameters()):
            s.data.mul_(self.decay).add_(m.data, alpha=1.0 - self.decay)

    def state_dict(self):   return self.shadow.state_dict()
    def load_state_dict(self, sd): self.shadow.load_state_dict(sd)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  GPUs: {torch.cuda.device_count()}")

    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    model = ConditionalUNet(base_ch=args.base_ch, dropout=args.dropout).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    raw       = model.module if isinstance(model, nn.DataParallel) else model
    ema       = EMA(raw)
    diffusion = GaussianDiffusion(T=args.T, device=str(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt        = torch.load(args.resume, map_location=device)
        raw.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}")

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for x, c in pbar:
            x, c = x.to(device), c.to(device)
            t    = torch.randint(0, args.T, (x.size(0),), device=device)
            loss = diffusion.p_losses(model, x, t, c, p_uncond=args.p_uncond)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1:4d}/{args.epochs}  loss {avg:.4f}  lr {scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            raw  = model.module if isinstance(model, nn.DataParallel) else model
            ckpt = {
                "epoch":    epoch + 1,
                "model":    raw.state_dict(),
                "ema":      ema.state_dict(),
                "optimizer":optimizer.state_dict(),
                "args":     vars(args),
            }
            path = os.path.join(args.save_dir, f"checkpoint_ep{epoch+1:04d}.pth")
            torch.save(ckpt, path)
            torch.save(ckpt, os.path.join(args.save_dir, "weights.pth"))
            print(f"  → saved {path}")

    print("Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train DDPM on CIFAR-10")
    p.add_argument("--data_dir",    default="./data")
    p.add_argument("--save_dir",    default=".")
    p.add_argument("--resume",      default=None)
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--epochs",      type=int,   default=500)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--T",           type=int,   default=1000)
    p.add_argument("--base_ch",     type=int,   default=128)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--p_uncond",    type=float, default=0.1)
    p.add_argument("--save_every",  type=int,   default=50)
    p.add_argument("--num_workers", type=int,   default=4)
    train(p.parse_args())
