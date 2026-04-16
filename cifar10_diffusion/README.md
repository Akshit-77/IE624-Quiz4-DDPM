# CIFAR-10 Diffusion Model (DDPM + DDIM)

IE 624 — Generative and Agentic AI | Quiz 4

---

## Architecture

| Component | Details |
|-----------|---------|
| Model | Class-conditional U-Net |
| Channels | 128 → 256 → 256 (at 32 → 16 → 8 px) |
| Attention | Self-attention at 16×16 and 8×8 |
| Conditioning | Timestep (sinusoidal) + class label (learned embedding) via Adaptive Group Norm |
| Classifier-Free Guidance | 10 % unconditional dropout during training |
| EMA | Exponential moving average (decay = 0.9999) for inference |
| Diffusion steps T | 1000 |
| Beta schedule | Linear, 1e-4 → 0.02 |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Training

### Local
```bash
python train.py \
    --epochs 500 \
    --batch_size 128 \
    --lr 2e-4 \
    --base_ch 128 \
    --save_dir ./checkpoints
```

### Modal (cloud GPU — A100)
```bash
pip install modal
modal setup          # one-time auth

# Start training (blocks until done)
modal run modal_train.py

# Custom hyperparameters
modal run modal_train.py --epochs 1000 --batch_size 256

# Download checkpoint after training
modal run modal_train.py --action download --output ./checkpoints/model.pth
```

---

## Inference

### Local

```bash
# DDIM sampling (fast, 50 steps) — class 3 (cat)
python inference.py --label_class 3 --sampler ddim

# DDPM sampling (full 1000 steps) — class 7 (horse)
python inference.py --label_class 7 --sampler ddpm

# More images, stronger guidance
python inference.py --label_class 5 --sampler ddim \
    --num_images 64 --guidance_scale 5.0 --ddim_steps 100

# Save every image individually
python inference.py --label_class 0 --sampler ddim --save_individual
```

#### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--label_class` | required | CIFAR-10 class 0-9 |
| `--sampler` | `ddim` | `ddpm` or `ddim` |
| `--checkpoint` | `./checkpoints/model.pth` | Path to `.pth` file |
| `--num_images` | `16` | Images to generate |
| `--guidance_scale` | `3.0` | CFG scale (1 = no CFG) |
| `--ddim_steps` | `50` | DDIM denoising steps |
| `--ddim_eta` | `0.0` | 0 = deterministic, 1 = DDPM stochastic |
| `--output_dir` | `./generated` | Where to save images |
| `--save_individual` | off | Also save each image separately |

---

### Modal (streamed — live VS Code preview)

Generates on a remote A100 and streams each batch back to your machine as it
finishes. VS Code image preview updates in real time.

```bash
# DDIM, class 3 (cat), 16 images (default)
modal run modal_inference.py --label_class 3

# DDPM, class 7 (horse), 32 images, guidance 5
modal run modal_inference.py --label_class 7 --sampler ddpm \
    --num_images 32 --guidance_scale 5.0

# All 10 classes, 16 images each, 8 at a time
modal run modal_inference.py --label_class -1 --num_images 16 --batch_size 8
```

Open `generated/grid_class<N>_<name>.png` in VS Code — it overwrites after
every batch so the preview refreshes live.

#### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--label_class` | `0` | CIFAR-10 class 0-9; `-1` = all classes |
| `--sampler` | `ddim` | `ddpm` or `ddim` |
| `--num_images` | `16` | Total images to generate |
| `--batch_size` | `4` | Images per streamed batch |
| `--guidance_scale` | `3.0` | CFG scale (1 = no CFG) |
| `--ddim_steps` | `50` | DDIM denoising steps |
| `--ddim_eta` | `0.0` | 0 = deterministic, 1 = DDPM stochastic |
| `--output_dir` | `./generated` | Local directory for saved images |
| `--checkpoint` | `""` | Override checkpoint path on volume |

---

## CIFAR-10 Class Labels

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | airplane | 5 | dog |
| 1 | automobile | 6 | frog |
| 2 | bird | 7 | horse |
| 3 | cat | 8 | ship |
| 4 | deer | 9 | truck |

---

## File Structure

```
cifar10_diffusion/
├── src/
│   ├── model.py           # U-Net noise predictor
│   └── diffusion.py       # DDPM / DDIM processes + CFG
├── train.py               # Training script with EMA
├── inference.py           # Local CLI image generation
├── modal_train.py         # Modal cloud training (A100)
├── modal_inference.py     # Modal streaming inference → live VS Code preview
├── requirements.txt
└── .gitignore
```
