"""
modal_inference.py — Streaming CIFAR-10 inference on a Modal A100 GPU.

Mounts the local `ddpm/` subdirectory into the container so the remote
function can import inference.py / model_train.py directly.

Images are generated in small batches on the GPU and streamed back one
batch at a time — VS Code image preview updates live as each arrives.

Usage:
    # DDIM, class 3 (cat), 16 images (default)
    modal run modal_inference.py --label_class 3

    # DDPM, class 7 (horse), 32 images, guidance 5
    modal run modal_inference.py --label_class 7 --sampler ddpm \
        --num_images 32 --guidance_scale 5.0

    # All 10 classes, 16 images each, streamed 4 at a time
    modal run modal_inference.py --label_class -1 --num_images 16 --batch_size 4

Open ddpm/generated/grid_class<N>_<name>.png in VS Code — it overwrites
after every batch so the preview refreshes live.

CIFAR-10 classes:
    0=airplane  1=automobile  2=bird  3=cat  4=deer
    5=dog       6=frog        7=horse 8=ship 9=truck
"""

import io
import math
from pathlib import Path

import modal

# ── App + persistent volume (same as training) ────────────────────────────────
app            = modal.App("cifar10-ddpm-v2")
vol            = modal.Volume.from_name("cifar10-ddpm-v2-vol", create_if_missing=True)
CHECKPOINT_DIR = "/vol/checkpoints"

# ── Container image — mounts local ddpm/ as /app/ddpm ────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .uv_pip_install("tqdm", "numpy", "Pillow")
    .add_local_dir("ddpm", "/app/ddpm")   # ← mounts the subdirectory
)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog",      "frog",       "horse", "ship", "truck",
]


# ── Remote generator — yields one PNG batch at a time ────────────────────────

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
    Yields (batch_index, label_class, png_bytes) one batch at a time.
    The caller saves each PNG immediately — no waiting for the full run.
    """
    import sys, os
    sys.path.append("/app/ddpm")        # make model_train / inference importable

    import torch
    from torchvision.utils import make_grid
    from PIL import Image as PILImage
    from model_train import ConditionalUNet, GaussianDiffusion

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[remote] device={device}  class={label_class}  "
          f"sampler={sampler.upper()}  total={num_images}  "
          f"batch={batch_size}  guidance={guidance_scale}")

    # Load checkpoint from volume (or override path)
    ckpt_path = checkpoint or os.path.join(CHECKPOINT_DIR, "weights.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Run modal_train.py first."
        )

    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    base_ch    = saved_args.get("base_ch", 128)
    T          = saved_args.get("T", 1000)

    model = ConditionalUNet(base_ch=base_ch, dropout=0.0)
    key   = "ema" if "ema" in ckpt else "model"
    model.load_state_dict(ckpt[key])
    model = model.to(device).eval()
    print(f"[remote] loaded {'EMA' if key == 'ema' else 'model'} weights")

    diffusion = GaussianDiffusion(T=T, device=str(device))

    generated = 0
    batch_idx = 0

    while generated < num_images:
        n     = min(batch_size, num_images - generated)
        shape = (n, 3, 32, 32)
        print(f"[remote] batch {batch_idx}: generating {n} image(s)…")

        if sampler == "ddpm":
            imgs = diffusion.ddpm_sample(model, shape, label_class, str(device), guidance_scale)
        else:
            imgs = diffusion.ddim_sample(model, shape, label_class, str(device),
                                         steps=ddim_steps, eta=ddim_eta,
                                         guidance_scale=guidance_scale)

        nrow = max(1, int(math.ceil(math.sqrt(n))))
        grid = make_grid(imgs, nrow=nrow, padding=2)
        arr  = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        buf  = io.BytesIO()
        PILImage.fromarray(arr).save(buf, format="PNG")
        png_bytes = buf.getvalue()

        print(f"[remote] batch {batch_idx}: done ({len(png_bytes)//1024} KB)")
        yield batch_idx, label_class, png_bytes

        generated += n
        batch_idx += 1


# ── Local entrypoint — receives streamed batches, saves to ddpm/generated/ ───

@app.local_entrypoint()
def main(
    label_class:    int   = -1,     # -1 = all 10 classes
    num_images:     int   = 16,
    sampler:        str   = "ddim",
    guidance_scale: float = 3.0,
    ddim_steps:     int   = 50,
    ddim_eta:       float = 0.0,
    batch_size:     int   = 4,
    output_dir:     str   = "./ddpm/generated",
    checkpoint:     str   = "",
):
    import numpy as np
    from PIL import Image as PILImage

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    classes_to_run = list(range(10)) if label_class == -1 else [label_class]

    for cls in classes_to_run:
        cls_name    = CIFAR10_CLASSES[cls]
        all_batches = []

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
            # Save individual batch grid (new file per batch)
            batch_path = out / f"batch_class{cls}_{cls_name}_{batch_idx:04d}.png"
            batch_path.write_bytes(png_bytes)
            print(f"  [local] batch {batch_idx:04d} → {batch_path}")

            # Rebuild running overview grid (overwrites each time → VS Code refreshes)
            all_batches.append(PILImage.open(io.BytesIO(png_bytes)))
            _update_grid(all_batches, out, cls, cls_name)

        print(f"  [local] done — {num_images} image(s) for class {cls} ({cls_name})")

    print(f"\nAll done. Browse results in {out}/")


def _update_grid(batch_images, out_dir, cls, cls_name):
    """Horizontally stitch all received batch PNGs into one overview file."""
    import numpy as np
    from PIL import Image as PILImage

    arrays = [np.array(img) for img in batch_images]
    max_h  = max(a.shape[0] for a in arrays)
    padded = []
    for a in arrays:
        if a.shape[0] < max_h:
            pad = np.ones((max_h - a.shape[0], a.shape[1], 3), dtype="uint8") * 255
            a   = np.vstack([a, pad])
        padded.append(a)

    overview  = np.hstack(padded)
    grid_path = out_dir / f"grid_class{cls}_{cls_name}.png"
    PILImage.fromarray(overview).save(grid_path)
    print(f"  [local] grid updated → {grid_path}")
