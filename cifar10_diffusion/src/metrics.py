"""
metrics.py — FID and IS computation for CIFAR-10 DDPM evaluation.

Uses torchmetrics for numerically reliable, GPU-accelerated computation.

  FID (Fréchet Inception Distance) — lower is better
      Measures distance between Inception feature distributions of
      generated vs. real images.  Requires a real-image reference set.

  IS  (Inception Score) — higher is better
      Measures sharpness + diversity of generated images alone.
      Does NOT require real images.

Minimum recommended sample sizes:
  IS  : ≥ 1 000 images  (50 000 for SOTA comparisons)
  FID : ≥ 1 000 images  (10 000–50 000 for SOTA comparisons)
"""

import sys
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

# Force line-buffered stdout so every print appears immediately in VS Code
sys.stdout.reconfigure(line_buffering=True)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_uint8(images: torch.Tensor) -> torch.Tensor:
    """Convert (N, 3, H, W) float [0, 1] → uint8 [0, 255]."""
    return (images.clamp(0.0, 1.0) * 255).to(torch.uint8)


def _load_cifar10(data_dir: str, train: bool = False,
                  n: int | None = None) -> torch.Tensor:
    """
    Load CIFAR-10 images as float [0, 1] tensors.

    Args:
        data_dir : directory to store / find CIFAR-10
        train    : use training split (True) or test split (False)
        n        : cap at this many images (None = full split)

    Returns:
        (N, 3, 32, 32) float tensor
    """
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=train, download=True, transform=T.ToTensor()
    )
    if n is not None:
        dataset = Subset(dataset, range(min(n, len(dataset))))

    loader = DataLoader(dataset, batch_size=512, num_workers=2, pin_memory=False)
    chunks = [x for x, _ in loader]
    return torch.cat(chunks, dim=0)   # (N, 3, 32, 32)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_inception_score(
    generated: torch.Tensor,
    device:    str = "cpu",
    splits:    int = 10,
) -> tuple[float, float]:
    """
    Compute Inception Score on generated images.

    Args:
        generated : (N, 3, H, W) float [0, 1]
        device    : 'cuda' or 'cpu'
        splits    : number of splits for variance estimation

    Returns:
        (mean, std) — higher mean is better
    """
    from torchmetrics.image.inception import InceptionScore

    metric = InceptionScore(splits=splits).to(device)
    metric.update(_to_uint8(generated).to(device))
    mean, std = metric.compute()
    return mean.item(), std.item()


def compute_fid(
    generated: torch.Tensor,
    real:      torch.Tensor,
    device:    str = "cpu",
) -> float:
    """
    Compute Fréchet Inception Distance between generated and real images.

    Args:
        generated : (N, 3, H, W) float [0, 1]
        real      : (M, 3, H, W) float [0, 1]
        device    : 'cuda' or 'cpu'

    Returns:
        FID score — lower is better
    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    metric.update(_to_uint8(real).to(device),      real=True)
    metric.update(_to_uint8(generated).to(device), real=False)
    return metric.compute().item()


def evaluate(
    generated:   torch.Tensor,
    data_dir:    str  = "./data",
    device:      str  = "cpu",
    n_real:      int  = 10_000,
    splits:      int  = 10,
    skip_fid:    bool = False,
) -> dict:
    """
    Compute IS (always) and FID (unless skip_fid=True) and print a summary.

    Args:
        generated : (N, 3, H, W) float [0, 1]  — generated images
        data_dir  : CIFAR-10 root (downloaded automatically if missing)
        device    : 'cuda' or 'cpu'
        n_real    : number of real CIFAR-10 images used for FID reference
        splits    : IS splits for variance estimation
        skip_fid  : set True to skip FID (e.g. when very few images)

    Returns:
        dict with keys 'IS_mean', 'IS_std', and optionally 'FID'
    """
    n = generated.shape[0]
    results: dict = {}

    if n < 100:
        print(f"\n  [metrics] WARNING: only {n} images — "
              "scores will have very high variance. Use ≥1 000 for reliable results.",
              flush=True)
    elif n < 1_000:
        print(f"\n  [metrics] NOTE: {n} images — "
              "scores are approximate. Use ≥1 000 for reliable FID/IS.",
              flush=True)

    # ── Inception Score ───────────────────────────────────────────────────────
    print("\n  [metrics] Computing Inception Score…", end=" ", flush=True)
    is_mean, is_std = compute_inception_score(generated, device=device, splits=splits)
    results["IS_mean"] = is_mean
    results["IS_std"]  = is_std
    print(f"done.  IS = {is_mean:.3f} ± {is_std:.3f}", flush=True)

    # ── FID ───────────────────────────────────────────────────────────────────
    if not skip_fid:
        print(f"  [metrics] Loading {n_real} real CIFAR-10 images…",
              end=" ", flush=True)
        real = _load_cifar10(data_dir, train=False, n=n_real)
        print(f"{real.shape[0]} loaded.", flush=True)

        print("  [metrics] Computing FID…", end=" ", flush=True)
        fid = compute_fid(generated, real, device=device)
        results["FID"] = fid
        print(f"done.  FID = {fid:.3f}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n  ┌─────────────────────────────────┐", flush=True)
    print(f"  │  IS   : {is_mean:7.3f} ± {is_std:.3f}          │", flush=True)
    if "FID" in results:
        print(f"  │  FID  : {results['FID']:7.3f}               │", flush=True)
    print("  └─────────────────────────────────┘", flush=True)

    return results
