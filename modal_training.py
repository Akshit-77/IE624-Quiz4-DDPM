"""
Modal wrapper for IE624 Quiz 4 — DDPM/DDIM Training
Run with:  modal run modal_training.py

After training completes, weights.pth is automatically downloaded
into IE624_531/weights.pth on your local machine.
"""

import os
import modal

ROLL      = "531"
LOCAL_DIR = f"IE624_{ROLL}"

app = modal.App("ie624-diffusion-training")

# ─── Container image ──────────────────────────────────────────────────────────
# add_local_file embeds training.py into the image (copy=True → built into layer)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .add_local_file(
        local_path=f"{LOCAL_DIR}/training.py",
        remote_path="/root/training.py",
        copy=True,
    )
)

# Modal volume — persists weights across runs
volume     = modal.Volume.from_name("ie624-weights", create_if_missing=True)
VOLUME_DIR = "/vol/weights"


# ─── Remote training function ─────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="A100",
    cpu=8,
    memory=32768,
    timeout=7200,           # 2 hours
    volumes={VOLUME_DIR: volume},
)
def train_diffusion():
    import subprocess
    import shutil

    print("=== IE624 Quiz 4 — Training on Modal A100 ===", flush=True)

    # Run training.py (saves weights.pth to /root/)
    result = subprocess.run(
        ["python", "/root/training.py"],
        cwd="/root",
        check=True,
    )

    # Copy final weights + periodic checkpoints to the volume
    import glob
    for pth in glob.glob("/root/weights*.pth"):
        dest = f"{VOLUME_DIR}/{os.path.basename(pth)}"
        shutil.copy(pth, dest)
        print(f"Copied {pth} -> {dest}", flush=True)

    volume.commit()
    return "Training complete — weights persisted to volume."


# ─── Local entrypoint ─────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    print(f"Submitting training job to Modal A100 ...")
    msg = train_diffusion.remote()
    print(f"Remote: {msg}")

    # Download weights.pth from volume to IE624_531/
    os.makedirs(LOCAL_DIR, exist_ok=True)
    vol = modal.Volume.from_name("ie624-weights")

    dest = os.path.join(LOCAL_DIR, "weights.pth")
    with open(dest, "wb") as f:
        f.write(b"".join(vol.read_file("weights.pth")))

    print(f"weights.pth downloaded to {dest}")
    print("Done. Submit IE624_531/ with training.py, inference.py, weights.pth.")
