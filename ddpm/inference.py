"""
inference.py — Generate CIFAR-10 images from a trained DDPM checkpoint.

Imports ConditionalUNet and GaussianDiffusion from model_train.py (same directory).

Usage:
    # DDIM, class 3 (cat), 16 images
    python inference.py --label_class 3

    # DDPM, class 7 (horse), 64 images, guidance 5
    python inference.py --label_class 7 --sampler ddpm --num_images 64 --guidance_scale 5.0

    # Save each image individually too
    python inference.py --label_class 0 --save_individual

CIFAR-10 classes:
    0=airplane  1=automobile  2=bird  3=cat  4=deer
    5=dog       6=frog        7=horse 8=ship 9=truck
"""

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from model_train import ConditionalUNet, GaussianDiffusion

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog",      "frog",       "horse", "ship", "truck",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    ckpt       = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    base_ch    = saved_args.get("base_ch", 128)

    model = ConditionalUNet(base_ch=base_ch, dropout=0.0)
    key   = "ema" if "ema" in ckpt else "model"
    model.load_state_dict(ckpt[key])
    print(f"Loaded {'EMA' if key == 'ema' else 'model'} weights from {checkpoint_path}")

    return model.to(device).eval(), saved_args


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate(args):
    if not 0 <= args.label_class <= 9:
        raise ValueError(f"--label_class must be 0-9, got {args.label_class}")

    device     = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    class_name = CIFAR10_CLASSES[args.label_class]

    print(f"Device         : {device}")
    print(f"Class          : {args.label_class} ({class_name})")
    print(f"Sampler        : {args.sampler.upper()}")
    print(f"Images         : {args.num_images}")
    print(f"Guidance scale : {args.guidance_scale}")

    model, saved_args = load_model(args.checkpoint, device)
    T         = saved_args.get("T", 1000)
    diffusion = GaussianDiffusion(T=T, device=str(device))
    shape     = (args.num_images, 3, 32, 32)

    print("Sampling…", end=" ", flush=True)
    if args.sampler == "ddpm":
        images = diffusion.ddpm_sample(
            model, shape, args.label_class, str(device), args.guidance_scale
        )
    else:
        images = diffusion.ddim_sample(
            model, shape, args.label_class, str(device),
            steps=args.ddim_steps, eta=args.ddim_eta, guidance_scale=args.guidance_scale,
        )
    print("done.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    import math
    nrow      = max(1, int(math.ceil(math.sqrt(args.num_images))))
    grid_path = out / f"class{args.label_class}_{class_name}_{args.sampler}.png"
    save_image(images, grid_path, nrow=nrow)
    print(f"Grid saved → {grid_path}")

    if args.save_individual:
        ind = out / f"class{args.label_class}_{class_name}"
        ind.mkdir(exist_ok=True)
        for i, img in enumerate(images):
            save_image(img, ind / f"{i:04d}.png")
        print(f"Individual images → {ind}/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate CIFAR-10 images from DDPM checkpoint")
    p.add_argument("--label_class",    type=int,   required=True,
                   help="Class 0-9: 0=airplane 1=auto 2=bird 3=cat 4=deer 5=dog 6=frog 7=horse 8=ship 9=truck")
    p.add_argument("--checkpoint",     type=str,   default="./weights.pth")
    p.add_argument("--sampler",        type=str,   default="ddim", choices=["ddpm", "ddim"])
    p.add_argument("--num_images",     type=int,   default=16)
    p.add_argument("--guidance_scale", type=float, default=3.0)
    p.add_argument("--ddim_steps",     type=int,   default=50)
    p.add_argument("--ddim_eta",       type=float, default=0.0)
    p.add_argument("--output_dir",     type=str,   default="./generated")
    p.add_argument("--save_individual",action="store_true")
    p.add_argument("--cpu",            action="store_true")
    generate(p.parse_args())
