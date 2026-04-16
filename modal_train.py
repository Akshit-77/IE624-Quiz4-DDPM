"""
modal_train.py — Train CIFAR-10 DDPM on a Modal A100 GPU.

Mounts the local `ddpm/` subdirectory into the container so the remote
function can import model_train.py directly.

Checkpoint is saved to a persistent Modal Volume and the final
weights.pth is downloaded back to ./ddpm/weights.pth on completion.

Usage:
    modal run modal_train.py                              # defaults
    modal run modal_train.py --epochs 1000 --base_ch 128
    modal run modal_train.py --epochs 500  --batch_size 256
"""

import modal

# ── App + persistent volume ───────────────────────────────────────────────────
app            = modal.App("cifar10-ddpm-v2")
vol            = modal.Volume.from_name("cifar10-ddpm-v2-vol", create_if_missing=True)
CHECKPOINT_DIR = "/vol/checkpoints"
DATA_DIR       = "/vol/data"

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


# ── Remote training function ──────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="A100",
    cpu=4,
    memory=12_000,
    timeout=3600 * 12,
    volumes={"/vol": vol},
)
def train_on_modal(
    batch_size: int   = 128,
    epochs:     int   = 500,
    lr:         float = 2e-4,
    base_ch:    int   = 128,
    save_every: int   = 50,
):
    import sys, os, argparse
    sys.path.append("/app/ddpm")        # make model_train importable
    from model_train import train

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,       exist_ok=True)

    resume = None
    weights_path = os.path.join(CHECKPOINT_DIR, "weights.pth")
    if os.path.isfile(weights_path):
        resume = weights_path
        print(f"Resuming from {resume}")

    args = argparse.Namespace(
        data_dir    = DATA_DIR,
        save_dir    = CHECKPOINT_DIR,
        resume      = resume,
        batch_size  = batch_size,
        epochs      = epochs,
        lr          = lr,
        T           = 1000,
        base_ch     = base_ch,
        dropout     = 0.1,
        p_uncond    = 0.1,
        save_every  = save_every,
        num_workers = 4,
    )
    train(args)
    vol.commit()
    print(f"Training complete. Checkpoints at {CHECKPOINT_DIR}")


# ── Local entrypoint — triggers training and downloads weights ────────────────

@app.local_entrypoint()
def main(
    batch_size: int   = 128,
    epochs:     int   = 500,
    lr:         float = 2e-4,
    base_ch:    int   = 128,
):
    import io
    from pathlib import Path

    print(f"Launching training: epochs={epochs}  batch={batch_size}  lr={lr}  base_ch={base_ch}")
    train_on_modal.remote(
        batch_size = batch_size,
        epochs     = epochs,
        lr         = lr,
        base_ch    = base_ch,
    )

    # Download weights.pth from the volume back to ddpm/
    print("\nDownloading weights.pth from volume…")
    dest = Path("ddpm") / "weights.pth"
    dest.parent.mkdir(exist_ok=True)

    with modal.Volume.from_name("cifar10-ddpm-v2-vol").batch_download() as batch:
        data = batch.get(f"{CHECKPOINT_DIR}/weights.pth")
        if data is not None:
            dest.write_bytes(data)
            print(f"Saved → {dest}  ({dest.stat().st_size // 1024} KB)")
        else:
            print("weights.pth not found on volume — check training completed successfully.")
