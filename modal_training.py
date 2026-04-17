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
        "torch",
        "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cu130",
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
    gpu="B200",
    cpu=8,
    memory=32768,
    timeout=7200,           # 2 hours
    volumes={VOLUME_DIR: volume},
)
def train_diffusion():
    import subprocess, os

    print("=== IE624 Quiz 4 — Training on Modal B200 ===", flush=True)

    # CKPT_DIR → volume path so checkpoints survive job restarts
    env = {**os.environ, "CKPT_DIR": VOLUME_DIR}

    subprocess.run(
        ["python", "/root/training.py"],
        cwd="/root",
        env=env,
        check=True,
    )

    # weights.pth is written to /root/ by training.py; copy to volume
    import shutil
    shutil.copy("/root/weights.pth", f"{VOLUME_DIR}/weights.pth")
    print(f"Copied weights.pth → {VOLUME_DIR}/weights.pth", flush=True)

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
