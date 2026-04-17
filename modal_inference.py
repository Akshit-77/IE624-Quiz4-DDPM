"""
Modal wrapper for IE624 Quiz 4 — Large-batch inference for FID/IS evaluation
Run with:  modal run modal_inference.py

Generates 1000 images per class (classes 0, 1, 2) on Modal GPU,
then downloads results to ./eval/ locally.

To test with a specific class:
  modal run modal_inference.py --label-class 1 --n-samples 1000
"""

import os
import modal

ROLL      = "531"
LOCAL_DIR = f"IE624_{ROLL}"

app = modal.App("ie624-diffusion-inference")

# ─── Container image ──────────────────────────────────────────────────────────
# add_local_file embeds inference.py and weights.pth into the image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "pytorch-fid",
        "torch-fidelity",
        "Pillow",
        "numpy",
    )
    .add_local_file(
        local_path=f"{LOCAL_DIR}/inference.py",
        remote_path="/root/inference.py",
        copy=True,
    )
    .add_local_file(
        local_path=f"{LOCAL_DIR}/weights.pth",
        remote_path="/root/weights.pth",
        copy=True,
    )
)

# Volume for output images
out_volume = modal.Volume.from_name("ie624-generated", create_if_missing=True)
OUT_DIR    = "/vol/generated"


# ─── Remote inference function ────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="L40S",
    cpu=4,
    memory=16384,
    timeout=3600,
    volumes={OUT_DIR: out_volume},
)
def run_inference(label_class: int = 1, n_samples: int = 1000, sampler: str = "ddim"):
    import subprocess

    out_path = f"{OUT_DIR}/class{label_class}"
    os.makedirs(out_path, exist_ok=True)

    print(
        f"Generating {n_samples} images for class {label_class} "
        f"using {sampler.upper()} ...",
        flush=True,
    )

    subprocess.run(
        [
            "python", "/root/inference.py",
            "--checkpoint_path",   "/root/weights.pth",
            "--label_class",       str(label_class),
            "--number_of_samples", str(n_samples),
            "--output_path",       out_path,
            "--sampler",           sampler,
        ],
        cwd="/root",
        check=True,
    )

    out_volume.commit()
    return f"Done: {n_samples} images for class {label_class} saved to {out_path}"


# ─── Local entrypoint ─────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(
    label_class: int = -1,   # -1 = run all three test classes (0, 1, 2)
    n_samples:   int = 1000,
    sampler:     str = "ddim",
):
    # Determine which classes to evaluate
    if label_class == -1:
        classes = [0, 1, 2]   # X-1, X, X+1 for roll 531
        print(f"Running inference for all three test classes: {classes}")
    else:
        classes = [label_class]
        print(f"Running inference for class {label_class}")

    # Submit jobs (can run in parallel — each is a separate Modal call)
    results = []
    for cls in classes:
        print(f"  Submitting class {cls} ...")
        msg = run_inference.remote(label_class=cls, n_samples=n_samples, sampler=sampler)
        results.append((cls, msg))

    for cls, msg in results:
        print(f"  Class {cls}: {msg}")

    # Download generated images from volume to ./eval/
    vol = modal.Volume.from_name("ie624-generated")
    local_eval = "./eval"
    os.makedirs(local_eval, exist_ok=True)

    for cls in classes:
        local_cls_dir = os.path.join(local_eval, f"class{cls}")
        os.makedirs(local_cls_dir, exist_ok=True)

        remote_prefix = f"class{cls}/"
        print(f"Downloading class {cls} images to {local_cls_dir} ...")

        for entry in vol.iterdir(remote_prefix):
            dest = os.path.join(local_cls_dir, os.path.basename(entry.path))
            with open(dest, "wb") as f:
                f.write(vol.read_file(entry.path))

        n_downloaded = len(os.listdir(local_cls_dir))
        print(f"  Downloaded {n_downloaded} files to {local_cls_dir}")

    print("\nAll done. Generated images are in ./eval/")
    print("Run pytorch-fid and torch-fidelity locally, or let inference.py print them.")
