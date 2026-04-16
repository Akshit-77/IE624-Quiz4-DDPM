"""
Training script for class-conditional DDPM on CIFAR-10.

Usage:
    python train.py                        # defaults: 500 epochs, batch 128
    python train.py --epochs 1000 --base_ch 128
    python train.py --resume checkpoints/model.pth
"""

import os
import copy
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

from src.model import ConditionalUNet
from src.diffusion import GaussianDiffusion


# ─────────────────────────────────────────────────────────────
# EMA helper
# ─────────────────────────────────────────────────────────────

class EMA:
    """Exponential moving average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay   = decay
        self.shadow  = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        raw = model.module if isinstance(model, nn.DataParallel) else model
        for s_p, m_p in zip(self.shadow.parameters(), raw.parameters()):
            s_p.data.mul_(self.decay).add_(m_p.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)


# ─────────────────────────────────────────────────────────────
# Training function (importable by modal_train.py)
# ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  GPUs: {torch.cuda.device_count()}")

    # ── Data ────────────────────────────────────────────────────
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),   # → [-1, 1]
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── Model ───────────────────────────────────────────────────
    model = ConditionalUNet(
        in_channels=3,
        base_ch=args.base_ch,
        num_classes=10,
        dropout=args.dropout,
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    ema = EMA(
        model.module if isinstance(model, nn.DataParallel) else model,
        decay=0.9999,
    )

    # ── Diffusion ───────────────────────────────────────────────
    diffusion = GaussianDiffusion(T=args.T, device=str(device))

    # ── Optimiser + scheduler ───────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Resume ──────────────────────────────────────────────────
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        raw  = model.module if isinstance(model, nn.DataParallel) else model
        raw.load_state_dict(ckpt["model_state_dict"])
        ema.load_state_dict(ckpt["ema_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Training loop ───────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for x, c in pbar:
            x = x.to(device)
            c = c.to(device)

            t = torch.randint(0, args.T, (x.size(0),), device=device)

            loss = diffusion.p_losses(
                model, x, t, c,
                p_uncond=args.p_uncond,
                null_class=10,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        avg = total_loss / len(loader)
        lr  = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:4d}/{args.epochs} | loss {avg:.4f} | lr {lr:.2e}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            raw = model.module if isinstance(model, nn.DataParallel) else model
            ckpt = {
                "epoch":               epoch + 1,
                "model_state_dict":    raw.state_dict(),
                "ema_state_dict":      ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss":                avg,
                "args":                vars(args),
            }
            path = os.path.join(args.save_dir, f"checkpoint_ep{epoch+1:04d}.pth")
            torch.save(ckpt, path)
            # Always keep a canonical model.pth pointing to the latest checkpoint
            torch.save(ckpt, os.path.join(args.save_dir, "model.pth"))
            print(f"  → saved {path}")

    print("Training complete.")


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

def get_parser():
    p = argparse.ArgumentParser(description="Train DDPM on CIFAR-10")
    p.add_argument("--data_dir",   type=str,   default="./data",        help="CIFAR-10 download dir")
    p.add_argument("--save_dir",   type=str,   default="./checkpoints", help="Checkpoint output dir")
    p.add_argument("--resume",     type=str,   default=None,            help="Resume from checkpoint")
    p.add_argument("--batch_size", type=int,   default=128)
    p.add_argument("--epochs",     type=int,   default=500)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--T",          type=int,   default=1000,            help="Number of diffusion steps")
    p.add_argument("--base_ch",    type=int,   default=128,             help="U-Net base channels")
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--p_uncond",   type=float, default=0.1,             help="CFG unconditional drop prob")
    p.add_argument("--save_every", type=int,   default=50,              help="Save every N epochs")
    p.add_argument("--num_workers",type=int,   default=4)
    return p


if __name__ == "__main__":
    args = get_parser().parse_args()
    train(args)
