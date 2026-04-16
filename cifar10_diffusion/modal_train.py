import modal

app = modal.App("cifar10-ddpm")

vol = modal.Volume.from_name("cifar10-ddpm-vol", create_if_missing=True)
CHECKPOINT_DIR = "/vol/checkpoints"
DATA_DIR       = "/vol/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .uv_pip_install("tqdm", "numpy", "Pillow")
    .add_local_dir("src", "/app/src")
    .add_local_dir(".", "/app", ignore=["src", ".git", "__pycache__", "*.pth", "data", "checkpoints", "generated", ".venv"])
)


@app.function(
    image=image,
    cpu=4,
    gpu="A100-80GB",
    memory=10000,
    timeout=3600 * 12,
    volumes={"/vol": vol},
)
def train_cifar10(
    batch_size: int   = 128,
    epochs:     int   = 500,
    lr:         float = 2e-4,
    base_ch:    int   = 128,
    save_every: int   = 50,
):
    import sys, os, argparse
    sys.path.append("/app")
    from train import train

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    resume = None
    if os.path.exists(f"{CHECKPOINT_DIR}/model.pth"):
        resume = f"{CHECKPOINT_DIR}/model.pth"
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
    print(f"Done — checkpoints at {CHECKPOINT_DIR}")


@app.function(
    image=image,
    volumes={"/vol": vol},
)
def read_checkpoint(remote_path: str) -> bytes:
    """Read a file from the volume and return its raw bytes."""
    with open(remote_path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(
    batch_size: int   = 128,
    epochs:     int   = 500,
    lr:         float = 2e-4,
    base_ch:    int   = 128,
    action:     str   = "train",   # "train" or "download"
    output:     str   = "./checkpoints/model.pth",
):
    if action == "download":
        import os
        remote_path = f"{CHECKPOINT_DIR}/model.pth"
        print(f"Downloading {remote_path} → {output} …")
        data = read_checkpoint.remote(remote_path)
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        with open(output, "wb") as f:
            f.write(data)
        print(f"Saved {len(data) // (1024*1024):.1f} MB → {output}")
        return

    train_cifar10.remote(
        batch_size = batch_size,
        epochs     = epochs,
        lr         = lr,
        base_ch    = base_ch,
    )
