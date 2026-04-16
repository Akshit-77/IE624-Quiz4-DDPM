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
):
    from PIL import Image as PILImage
    import numpy as np

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    classes_to_run = list(range(10)) if label_class == -1 else [label_class]

    for cls in classes_to_run:
        cls_name    = CIFAR10_CLASSES[cls]
        all_batches = []   # collect PIL images for the running grid

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
            # ── Save individual batch grid ───────────────────────────────
            batch_path = out / f"batch_class{cls}_{cls_name}_{batch_idx:04d}.png"
            batch_path.write_bytes(png_bytes)
            print(f"  [local] batch {batch_idx:04d} saved → {batch_path}")

            # ── Append to running collection & rebuild full grid ─────────
            all_batches.append(PILImage.open(io.BytesIO(png_bytes)))

            _save_running_grid(all_batches, out, cls, cls_name)

        print(f"  [local] class {cls} ({cls_name}) complete — "
              f"{num_images} image(s) in {out}/")

    print(f"\nAll done. Open {out}/ in VS Code Explorer to browse results.")


# ── Helper: stitch all batch-grid PNGs into one growing overview ─────────────

def _save_running_grid(
    batch_images: list,
    out_dir:      Path,
    cls:          int,
    cls_name:     str,
):
    """Horizontally stack all batch grids and save as a single overview PNG."""
    from PIL import Image as PILImage
    import numpy as np

    arrays = [np.array(img) for img in batch_images]
    # Pad heights to match (they should be equal, but be safe)
    max_h = max(a.shape[0] for a in arrays)
    padded = []
    for a in arrays:
        if a.shape[0] < max_h:
            pad = np.ones((max_h - a.shape[0], a.shape[1], 3), dtype=np.uint8) * 255
            a = np.vstack([a, pad])
        padded.append(a)

    overview = np.hstack(padded)
    grid_path = out_dir / f"grid_class{cls}_{cls_name}.png"
    PILImage.fromarray(overview).save(grid_path)
    print(f"  [local] running grid updated → {grid_path}")
