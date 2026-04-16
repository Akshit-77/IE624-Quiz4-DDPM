"""
Inference script for CIFAR-10 DDPM / DDIM image generation.

Usage examples:
    # DDPM sampling, class 3 (cat), 16 images
    python inference.py --label_class 3 --sampler ddpm

    # DDIM sampling (50 steps), class 7 (horse), 64 images, guidance 5
    python inference.py --label_class 7 --sampler ddim --num_images 64 --guidance_scale 5.0

    # Use a specific checkpoint
    python inference.py --label_class 0 --sampler ddim --checkpoint ./checkpoints/model.pth

CIFAR-10 class map:
    0=airplane  1=automobile  2=bird  3=cat  4=deer
    5=dog       6=frog        7=horse 8=ship 9=truck
"""

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from src.model import ConditionalUNet
from src.diffusion import GaussianDiffusion


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Load EMA weights (preferred) from a .pth checkpoint."""
    ckpt      = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    base_ch   = saved_args.get("base_ch", 128)

    model = ConditionalUNet(
        in_channels=3,
        base_ch=base_ch,
        num_classes=10,
        dropout=0.0,          # no dropout at inference
    )

    if "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
        print("Loaded EMA weights.")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        print("Loaded model weights (no EMA found).")

    model = model.to(device).eval()
    return model, saved_args


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def generate(args):
    if not 0 <= args.label_class <= 9:
        raise ValueError(f"--label_class must be in [0, 9], got {args.label_class}")

    device     = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    class_name = CIFAR10_CLASSES[args.label_class]

    print(f"Device        : {device}")
    print(f"Class         : {args.label_class} ({class_name})")
    print(f"Sampler       : {args.sampler.upper()}")
    print(f"Num images    : {args.num_images}")
    print(f"Guidance scale: {args.guidance_scale}")

    # ── Load model ──────────────────────────────────────────────
    model, saved_args = load_model(args.checkpoint, device)
    T = saved_args.get("T", 1000)
    diffusion = GaussianDiffusion(T=T, device=str(device))

    shape = (args.num_images, 3, 32, 32)

    # ── Sample ──────────────────────────────────────────────────
    print(f"\nSampling…", end=" ", flush=True)

    if args.sampler.lower() == "ddpm":
        images = diffusion.ddpm_sample(
            model,
            shape        = shape,
            label_class  = args.label_class,
            device       = str(device),
            guidance_scale = args.guidance_scale,
        )
    elif args.sampler.lower() == "ddim":
        images = diffusion.ddim_sample(
            model,
            shape        = shape,
            label_class  = args.label_class,
            device       = str(device),
            steps        = args.ddim_steps,
            eta          = args.ddim_eta,
            guidance_scale = args.guidance_scale,
        )
    else:
        raise ValueError(f"Unknown sampler '{args.sampler}'. Choose 'ddpm' or 'ddim'.")

    print("done.")

    # ── Save ────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Grid image
    nrow      = max(1, int(args.num_images ** 0.5))
    grid_path = out_dir / f"class{args.label_class}_{class_name}_{args.sampler}.png"
    save_image(images, grid_path, nrow=nrow)
    print(f"Saved grid  → {grid_path}")

    # Individual images (optional)
    if args.save_individual:
        ind_dir = out_dir / f"class{args.label_class}_{class_name}_{args.sampler}"
        ind_dir.mkdir(exist_ok=True)
        for i, img in enumerate(images):
            save_image(img, ind_dir / f"{i:04d}.png")
        print(f"Saved {len(images)} images → {ind_dir}/")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def get_parser():
    p = argparse.ArgumentParser(
        description="Generate CIFAR-10 images with DDPM or DDIM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--label_class", type=int, required=True,
        help="CIFAR-10 class [0-9]: 0=airplane 1=auto 2=bird 3=cat 4=deer "
             "5=dog 6=frog 7=horse 8=ship 9=truck",
    )
    p.add_argument(
        "--sampler", type=str, default="ddim", choices=["ddpm", "ddim"],
        help="Sampling procedure (default: ddim)",
    )
    p.add_argument(
        "--checkpoint", type=str, default="./checkpoints/model.pth",
        help="Path to model checkpoint (.pth)",
    )
    p.add_argument(
        "--num_images", type=int, default=16,
        help="Number of images to generate (default: 16)",
    )
    p.add_argument(
        "--output_dir", type=str, default="./generated",
        help="Output directory (default: ./generated)",
    )
    p.add_argument(
        "--guidance_scale", type=float, default=3.0,
        help="Classifier-free guidance scale (1.0 = no CFG, default: 3.0)",
    )
    p.add_argument(
        "--ddim_steps", type=int, default=50,
        help="Number of DDIM denoising steps (default: 50)",
    )
    p.add_argument(
        "--ddim_eta", type=float, default=0.0,
        help="DDIM η: 0=deterministic, 1=DDPM stochastic (default: 0.0)",
    )
    p.add_argument(
        "--save_individual", action="store_true",
        help="Also save each image individually",
    )
    p.add_argument(
        "--cpu", action="store_true",
        help="Force CPU inference",
    )
    return p


if __name__ == "__main__":
    args = get_parser().parse_args()
    generate(args)
