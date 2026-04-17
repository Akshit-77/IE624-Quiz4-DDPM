"""
IE624 Quiz 4 — Inference Script
Roll: 531  |  Test classes: 0, 1, 2

Usage:
  python inference.py \
    --checkpoint_path weights.pth \
    --label_class 1 \
    --number_of_samples 100 \
    --output_path /path/to/output
"""

import os
import math
import shutil
import argparse
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────
NUM_CLASSES = 10
NULL_CLASS  = 10     # CFG null token
T           = 1000
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


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
        betas               = cosine_beta_schedule(T).to(device)
        alphas              = 1.0 - betas
        alphas_cumprod      = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.T                             = T
        self.betas                         = betas
        self.alphas                        = alphas
        self.alphas_cumprod                = alphas_cumprod
        self.alphas_cumprod_prev           = alphas_cumprod_prev
        self.sqrt_alphas_cumprod           = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod).sqrt()
        self.sqrt_recip_alphas             = alphas.rsqrt()
        self.posterior_variance            = (
            betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )


# ─── U-Net (identical to training.py) ────────────────────────────────────────

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


class UNet(nn.Module):
    def __init__(self, base_ch=128, num_classes=NUM_CLASSES):
        super().__init__()
        n_cls    = num_classes + 1
        time_dim = base_ch * 4
        ch       = [base_ch, base_ch * 2, base_ch * 4]

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv  = nn.Conv2d(3, ch[0], 3, padding=1)

        self.down1a     = ResBlock(ch[0], ch[0], time_dim, n_cls)
        self.down1b     = ResBlock(ch[0], ch[0], time_dim, n_cls)
        self.down1_ds   = Downsample(ch[0])

        self.down2a     = ResBlock(ch[0], ch[1], time_dim, n_cls)
        self.down2b     = ResBlock(ch[1], ch[1], time_dim, n_cls)
        self.attn_down2 = SelfAttention(ch[1])
        self.down2_ds   = Downsample(ch[1])

        self.down3a     = ResBlock(ch[1], ch[2], time_dim, n_cls)
        self.down3b     = ResBlock(ch[2], ch[2], time_dim, n_cls)
        self.down3_ds   = Downsample(ch[2])

        self.mid1       = ResBlock(ch[2], ch[2], time_dim, n_cls)
        self.mid_attn   = SelfAttention(ch[2])
        self.mid2       = ResBlock(ch[2], ch[2], time_dim, n_cls)

        self.up3_us     = Upsample(ch[2])
        self.up3a       = ResBlock(ch[2] + ch[2], ch[2], time_dim, n_cls)
        self.up3b       = ResBlock(ch[2], ch[2], time_dim, n_cls)

        self.up2_us     = Upsample(ch[2])
        self.up2a       = ResBlock(ch[2] + ch[1], ch[1], time_dim, n_cls)
        self.up2b       = ResBlock(ch[1], ch[1], time_dim, n_cls)
        self.attn_up2   = SelfAttention(ch[1])

        self.up1_us     = Upsample(ch[1])
        self.up1a       = ResBlock(ch[1] + ch[0], ch[0], time_dim, n_cls)
        self.up1b       = ResBlock(ch[0], ch[0], time_dim, n_cls)

        self.out_norm   = nn.GroupNorm(8, ch[0])
        self.out_conv   = nn.Conv2d(ch[0], 3, 1)

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


# ─── DDPM Sampler ─────────────────────────────────────────────────────────────

@torch.no_grad()
def ddpm_sample(model, dc, n_samples, label, cfg_scale=3.0, batch_size=32):
    """Full T-step DDPM reverse diffusion with classifier-free guidance."""
    model.eval()
    all_imgs  = []
    remaining = n_samples

    while remaining > 0:
        bs           = min(batch_size, remaining)
        x            = torch.randn(bs, 3, 32, 32, device=DEVICE)
        cond_labels  = torch.full((bs,), label,      dtype=torch.long, device=DEVICE)
        null_labels  = torch.full((bs,), NULL_CLASS, dtype=torch.long, device=DEVICE)

        for t_idx in reversed(range(dc.T)):
            t_batch    = torch.full((bs,), t_idx, device=DEVICE, dtype=torch.long)

            eps_cond   = model(x, t_batch, cond_labels)
            eps_uncond = model(x, t_batch, null_labels)
            eps        = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

            beta     = dc.betas[t_idx]
            sqrt_1ma = dc.sqrt_one_minus_alphas_cumprod[t_idx]
            mean     = dc.sqrt_recip_alphas[t_idx] * (x - beta / sqrt_1ma * eps)

            if t_idx > 0:
                noise = torch.randn_like(x)
                x     = mean + dc.posterior_variance[t_idx].sqrt() * noise
            else:
                x = mean

        all_imgs.append(x.cpu())
        remaining -= bs

    return torch.cat(all_imgs, dim=0)[:n_samples]


# ─── DDIM Sampler ─────────────────────────────────────────────────────────────

@torch.no_grad()
def ddim_sample(model, dc, n_samples, label, steps=50, eta=0.0, cfg_scale=3.0, batch_size=32):
    """DDIM sampler (deterministic by default, eta=0) with CFG."""
    model.eval()
    # Evenly spaced timesteps descending from T-1 to 0
    timesteps = torch.linspace(dc.T - 1, 0, steps + 1, dtype=torch.long)

    all_imgs  = []
    remaining = n_samples

    while remaining > 0:
        bs          = min(batch_size, remaining)
        x           = torch.randn(bs, 3, 32, 32, device=DEVICE)
        cond_labels = torch.full((bs,), label,      dtype=torch.long, device=DEVICE)
        null_labels = torch.full((bs,), NULL_CLASS, dtype=torch.long, device=DEVICE)

        for i in range(steps):
            t      = int(timesteps[i].item())
            t_prev = int(timesteps[i + 1].item())

            t_batch    = torch.full((bs,), t, device=DEVICE, dtype=torch.long)
            eps_cond   = model(x, t_batch, cond_labels)
            eps_uncond = model(x, t_batch, null_labels)
            eps        = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

            alpha_bar      = dc.alphas_cumprod[t]
            alpha_bar_prev = dc.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=DEVICE)

            # Predict x0
            x0_pred = (x - (1 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)

            # DDIM update
            sigma   = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar) *
                             (1 - alpha_bar / alpha_bar_prev)).sqrt()
            dir_xt  = (1 - alpha_bar_prev - sigma ** 2).sqrt() * eps
            noise   = sigma * torch.randn_like(x) if eta > 0.0 else 0.0

            x = alpha_bar_prev.sqrt() * x0_pred + dir_xt + noise

        all_imgs.append(x.cpu())
        remaining -= bs

    return torch.cat(all_imgs, dim=0)[:n_samples]


# ─── Metrics ──────────────────────────────────────────────────────────────────

def save_real_cifar_images(label_class, out_dir):
    """Pull CIFAR-10 test split, filter to label_class, save as PNG."""
    os.makedirs(out_dir, exist_ok=True)
    ds = datasets.CIFAR10(
        root="/tmp/cifar10_ref", train=False, download=True,
        transform=transforms.ToTensor(),
    )
    count = 0
    for img, lbl in ds:
        if lbl == label_class:
            arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(out_dir, f"real_{count:05d}.png"))
            count += 1
    return count


def compute_metrics(gen_dir, real_dir):
    from pytorch_fid.fid_score import calculate_fid_given_paths
    import torch_fidelity

    fid = calculate_fid_given_paths(
        [real_dir, gen_dir],
        batch_size=64,
        device=DEVICE,
        dims=2048,
    )

    m = torch_fidelity.calculate_metrics(
        input1=gen_dir,
        cuda=(DEVICE == "cuda"),
        isc=True,
        fid=False,
        verbose=False,
    )
    return fid, m["inception_score_mean"], m["inception_score_std"]


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IE624 Quiz 4 — Diffusion Inference")
    parser.add_argument("--checkpoint_path",  required=True,  type=str,
                        help="Path to weights.pth")
    parser.add_argument("--label_class",       required=True,  type=int,
                        help="CIFAR-10 class label 0-9")
    parser.add_argument("--number_of_samples", required=True,  type=int,
                        help="Number of images to generate")
    parser.add_argument("--output_path",       required=True,  type=str,
                        help="Directory to save generated PNG images")
    parser.add_argument("--sampler",           default="ddim",
                        choices=["ddpm", "ddim"],
                        help="Sampling method (default: ddim)")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"Loading weights from {args.checkpoint_path} ...", flush=True)
    model = UNet().to(DEVICE)
    state = torch.load(args.checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded on {DEVICE}.", flush=True)

    dc = DiffusionConstants(T=T, device=DEVICE)

    # ── Generate ───────────────────────────────────────────────────────────────
    print(
        f"Generating {args.number_of_samples} images "
        f"for class {args.label_class} using {args.sampler.upper()} ...",
        flush=True,
    )

    if args.sampler == "ddpm":
        imgs = ddpm_sample(model, dc, args.number_of_samples, args.label_class)
    else:
        imgs = ddim_sample(model, dc, args.number_of_samples, args.label_class)

    # ── Save as uint8 PNG ──────────────────────────────────────────────────────
    for i, img in enumerate(imgs):
        arr = ((img.clamp(-1, 1) + 1) * 0.5 * 255).permute(1, 2, 0).byte().numpy()
        Image.fromarray(arr.astype(np.uint8)).save(
            os.path.join(args.output_path, f"sample_{i:05d}.png")
        )

    print(f"Images saved to: {args.output_path}", flush=True)

    # ── FID / IS ───────────────────────────────────────────────────────────────
    tmp_real = "/tmp/cifar10_real_ref_cls"
    try:
        n_real = save_real_cifar_images(args.label_class, tmp_real)
        print(
            f"Reference: {n_real} real CIFAR-10 test images for class {args.label_class}",
            flush=True,
        )

        if args.number_of_samples < 1000:
            print(
                f"[WARNING] Only {args.number_of_samples} samples generated. "
                "FID is statistically unreliable below 1000. "
                "Scores below are indicative only.",
                flush=True,
            )

        fid, is_mean, is_std = compute_metrics(args.output_path, tmp_real)

        print("", flush=True)
        print("------------------------------------", flush=True)
        print(f"FID Score  : {fid:.2f}", flush=True)
        print(f"IS Score   : {is_mean:.2f} ± {is_std:.2f}", flush=True)
        print("------------------------------------", flush=True)

    finally:
        if os.path.exists(tmp_real):
            shutil.rmtree(tmp_real)


if __name__ == "__main__":
    main()
