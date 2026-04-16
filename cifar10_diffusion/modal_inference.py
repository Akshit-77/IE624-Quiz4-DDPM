"""
Streaming inference on Modal for CIFAR-10 DDPM/DDIM.

Generates images in small batches on a remote GPU and streams each batch
back to your local machine as it finishes — so VS Code image preview
updates live rather than waiting for the full run.

Usage:
    # DDIM (fast), all 10 classes, 16 images each
    modal run modal_inference.py

    # DDPM, class 3 (cat), 32 images, guidance 5
    modal run modal_inference.py --label_class 3 --sampler ddpm --num_images 32 --guidance_scale 5.0

    # DDIM, class 7 (horse), 64 images, batch 8 at a time
    modal run modal_inference.py --label_class 7 --num_images 64 --batch_size 8

    # All classes, 16 images each
    modal run modal_inference.py --label_class -1 --num_images 16

CIFAR-10 class map:
    0=airplane  1=automobile  2=bird  3=cat  4=deer
    5=dog       6=frog        7=horse 8=ship 9=truck

Streamed output:
    generated/
      batch_class3_cat_0000.png   ← individual batch grids (appear as they finish)
      batch_class3_cat_0001.png
      ...
      grid_class3_cat.png         ← running full grid (overwrites each batch)
"""

import io
import math
from pathlib import Path

import modal

# ── Reuse the same app / volume as training ──────────────────────────────────
app            = modal.App("cifar10-ddpm")
vol            = modal.Volume.from_name("cifar10-ddpm-vol", create_if_missing=True)
CHECKPOINT_DIR = "/vol/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .uv_pip_install("tqdm", "numpy", "Pillow")
    .uv_pip_install("torchmetrics[image]", "scipy")
    .add_local_dir("src", "/app/src")
    .add_local_dir(
        ".", "/app",
        ignore=["src", ".git", "__pycache__", "*.pth", "data",
                "checkpoints", "generated", ".venv"],
    )
)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog",      "frog",       "horse", "ship", "truck",
]


# ── Remote generator function ────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="A100",
    timeout=600,
    volumes={"/vol": vol},
)
def stream_generate(
    label_class:    int   = 0,
    num_images:     int   = 16,
    sampler:        str   = "ddim",
    guidance_scale: float = 3.0,
    ddim_steps:     int   = 50,
    ddim_eta:       float = 0.0,
    batch_size:     int   = 4,
    checkpoint:     str   = "",
):
    """
    Yields (batch_index, label_class, png_bytes) tuples one batch at a time.

    Each yielded item is a PNG-encoded grid of `batch_size` images so the
    caller can save / display it immediately without waiting for the full run.
    """
    import sys, os
    sys.path.append("/app")

    import torch
    from torchvision.utils import make_grid
    from PIL import Image as PILImage

    from src.model     import ConditionalUNet
    from src.diffusion import GaussianDiffusion

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[remote] device={device}  class={label_class}  sampler={sampler.upper()}"
          f"  total={num_images}  batch={batch_size}  guidance={guidance_scale}")

    # ── Load checkpoint ──────────────────────────────────────────────────────
    ckpt_path = checkpoint or os.path.join(CHECKPOINT_DIR, "model.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            "Make sure training has completed and the volume is populated."
        )

    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    base_ch    = saved_args.get("base_ch", 128)
    T          = saved_args.get("T", 1000)

    model = ConditionalUNet(in_channels=3, base_ch=base_ch, num_classes=10, dropout=0.0)
    if "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
        print("[remote] Loaded EMA weights.")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        print("[remote] Loaded model weights (no EMA found).")
    model = model.to(device).eval()

    diffusion = GaussianDiffusion(T=T, device=str(device))

    # ── Generate batch by batch, yielding PNG bytes each time ───────────────
    generated = 0
    batch_idx = 0

    while generated < num_images:
        current_batch = min(batch_size, num_images - generated)
        shape = (current_batch, 3, 32, 32)

        print(f"[remote] batch {batch_idx}: generating {current_batch} image(s) …")

        if sampler.lower() == "ddpm":
            imgs = diffusion.ddpm_sample(
                model,
                shape          = shape,
                label_class    = label_class,
                device         = str(device),
                guidance_scale = guidance_scale,
            )
        else:
            imgs = diffusion.ddim_sample(
                model,
                shape          = shape,
                label_class    = label_class,
                device         = str(device),
                steps          = ddim_steps,
                eta            = ddim_eta,
                guidance_scale = guidance_scale,
            )

        # Encode batch as a PNG grid in memory
        nrow  = max(1, int(math.ceil(math.sqrt(current_batch))))
        grid  = make_grid(imgs, nrow=nrow, padding=2)           # (3, H, W) in [0,1]
        arr   = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        buf   = io.BytesIO()
        PILImage.fromarray(arr).save(buf, format="PNG")
        png_bytes = buf.getvalue()

        print(f"[remote] batch {batch_idx}: done ({len(png_bytes)//1024} KB)")
        yield batch_idx, label_class, png_bytes

        generated += current_batch
        batch_idx += 1


# ── Remote metrics function — runs entirely on Modal GPU ─────────────────────

@app.function(
    image=image,
    gpu="A100",
    timeout=1800,
    volumes={"/vol": vol},
)
def evaluate_metrics(
    num_samples:    int   = 1000,
    label_class:    int   = -1,      # -1 = all classes equally
    sampler:        str   = "ddim",
    guidance_scale: float = 3.0,
    ddim_steps:     int   = 50,
    ddim_eta:       float = 0.0,
    n_real:         int   = 10_000,
    checkpoint:     str   = "",
) -> dict:
    """
    Generate `num_samples` images on the GPU, then compute FID and IS
    against the CIFAR-10 test set — all on the remote server.

    Returns dict: {'IS_mean', 'IS_std', 'FID', 'num_generated', 'num_real'}
    """
    import sys, os
    sys.path.append("/app")
    sys.path.append("/app/src")

    import torch
    from src.model     import ConditionalUNet
    from src.diffusion import GaussianDiffusion
    from src.metrics   import evaluate

    device = torch.device("cuda")
    print(f"[metrics] device={device}  samples={num_samples}  "
          f"sampler={sampler.upper()}  guidance={guidance_scale}")

    # ── Load model ───────────────────────────────────────────────────────────
    ckpt_path = checkpoint or os.path.join(CHECKPOINT_DIR, "model.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train first.")

    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    base_ch    = saved_args.get("base_ch", 128)
    T          = saved_args.get("T", 1000)

    model = ConditionalUNet(in_channels=3, base_ch=base_ch, num_classes=10, dropout=0.0)
    key   = "ema_state_dict" if "ema_state_dict" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[key])
    model = model.to(device).eval()
    print(f"[metrics] loaded {'EMA' if 'ema' in key else 'model'} weights")

    diffusion = GaussianDiffusion(T=T, device=str(device))

    # ── Generate images ───────────────────────────────────────────────────────
    classes    = list(range(10)) if label_class == -1 else [label_class]
    per_class  = num_samples // len(classes)
    all_images = []

    for cls in classes:
        remaining = num_samples - len(all_images)
        n         = min(per_class, remaining)
        if n <= 0:
            break
        print(f"[metrics] generating {n} images for class {cls}…")
        shape = (n, 3, 32, 32)
        if sampler == "ddpm":
            imgs = diffusion.ddpm_sample(model, shape, cls, str(device), guidance_scale)
        else:
            imgs = diffusion.ddim_sample(model, shape, cls, str(device),
                                         steps=ddim_steps, eta=ddim_eta,
                                         guidance_scale=guidance_scale)
        all_images.append(imgs.cpu())

    generated = torch.cat(all_images, dim=0)
    print(f"[metrics] total generated: {generated.shape[0]}")

    # ── Compute FID + IS ──────────────────────────────────────────────────────
    results = evaluate(
        generated = generated,
        data_dir  = "/vol/data",
        device    = str(device),
        n_real    = n_real,
    )
    results["num_generated"] = generated.shape[0]
    results["num_real"]      = n_real
    return results


# ── Local entrypoint — streams results to disk as they arrive ────────────────

@app.local_entrypoint()
def main(
    label_class:    int   = -1,     # -1 = all 10 classes
    num_images:     int   = 16,
    sampler:        str   = "ddim",
    guidance_scale: float = 3.0,
    ddim_steps:     int   = 50,
    ddim_eta:       float = 0.0,
    batch_size:     int   = 4,
    output_dir:     str   = "./generated",
    checkpoint:     str   = "",
    compute_metrics: bool = False,  # pass --compute-metrics to enable
    metrics_samples: int  = 1000,   # images to generate for metrics
):
    # No local PIL/numpy needed — remote sends ready-made PNG bytes
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    classes_to_run = list(range(10)) if label_class == -1 else [label_class]

    for cls in classes_to_run:
        cls_name   = CIFAR10_CLASSES[cls]
        batch_paths = []

        print(f"\n── Class {cls} ({cls_name}) ──────────────────────────────")

        for batch_idx, _, png_bytes in stream_generate.remote_gen(
            label_class    = cls,
            num_images     = num_images,
            sampler        = sampler,
            guidance_scale = guidance_scale,
            ddim_steps     = ddim_steps,
            ddim_eta       = ddim_eta,
            batch_size     = batch_size,
            checkpoint     = checkpoint,
        ):
            # Write PNG bytes directly — no local PIL needed
            batch_path = out / f"batch_class{cls}_{cls_name}_{batch_idx:04d}.png"
            batch_path.write_bytes(png_bytes)
            batch_paths.append(batch_path)
            print(f"  [local] batch {batch_idx:04d} saved → {batch_path}")

            # Try to build a running grid if Pillow happens to be installed
            _try_update_grid(batch_paths, out, cls, cls_name)

        print(f"  [local] class {cls} ({cls_name}) complete — "
              f"{num_images} image(s) in {out}/")

    # ── FID / IS (runs on Modal GPU, reports back) ────────────────────────────
    if compute_metrics:
        print(f"\n── Computing FID & IS on Modal GPU "
              f"({metrics_samples} samples) ────────────")
        scores = evaluate_metrics.remote(
            num_samples    = metrics_samples,
            label_class    = label_class,
            sampler        = sampler,
            guidance_scale = guidance_scale,
            ddim_steps     = ddim_steps,
            ddim_eta       = ddim_eta,
            checkpoint     = checkpoint,
        )
        print(f"\n  ┌─────────────────────────────────────────┐")
        print(f"  │  Samples : {scores['num_generated']:>6}  "
              f"Real ref : {scores['num_real']:>6}      │")
        print(f"  │  IS      : {scores['IS_mean']:>7.3f} ± {scores['IS_std']:.3f}          │")
        print(f"  │  FID     : {scores['FID']:>7.3f}                       │")
        print(f"  └─────────────────────────────────────────┘")

    print(f"\nAll done. Open {out}/ in VS Code Explorer to browse results.")


# ── Optional helper: stitch batch PNGs into one growing overview ──────────────

def _try_update_grid(
    batch_paths: list,
    out_dir:     Path,
    cls:         int,
    cls_name:    str,
):
    """
    Horizontally stack saved batch PNGs into a single overview file.
    Silently skipped if Pillow is not installed locally.
    """
    try:
        from PIL import Image as PILImage
    except ImportError:
        return   # Pillow not installed locally — individual batch files still saved

    images  = [PILImage.open(p) for p in batch_paths]
    max_h   = max(img.height for img in images)
    widths  = [img.width for img in images]
    canvas  = PILImage.new("RGB", (sum(widths), max_h), (255, 255, 255))
    x       = 0
    for img in images:
        canvas.paste(img, (x, 0))
        x += img.width

    grid_path = out_dir / f"grid_class{cls}_{cls_name}.png"
    canvas.save(grid_path)
    print(f"  [local] running grid updated → {grid_path}")
