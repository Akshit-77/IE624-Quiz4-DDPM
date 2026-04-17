# IE624 Quiz 4 — Execution Plan (Final)
# the XXX placeholder is 531 everywhere, so take action accordingly
---

## Directory Structure

```
IE624_XXX/                     ← XXX = your roll number; this gets zipped as IE624_XXX.zip
├── training.py
├── inference.py
└── weights.pth                ← downloaded after training completes on Modal
```

**Working directory during development (not submitted):**
```
quiz4/
├── modal_training.py          ← Modal wrapper for training
├── modal_inference.py         ← Modal wrapper for inference
└── IE624_XXX/
    ├── training.py
    ├── inference.py
    └── weights.pth
```

---

## How the Evaluator Will Run Your Code

```bash
python inference.py \
  --checkpoint_path "\path_as_user_input\weights.pth" \
  --label_class 3 \
  --number_of_samples 100 \
  --output_path "/content/your_roll_number/class/images"
```

Every design decision flows from making this command work perfectly. The inference script must accept exactly these four arguments and nothing else should be required from the evaluator.

---

## Phase 1 — Build the Core Model

Write the U-Net and all supporting components **directly inside `training.py`**. Since there are only three files to submit, there is no separate model file — everything lives in the two scripts.

Both `training.py` and `inference.py` will contain the **same model class definition**. This is intentional and necessary — do not try to import across files.

Core components to implement inside both scripts:

- U-Net with ResBlocks, Self-Attention at 16×16, time embedding, class conditioning
- Cosine noise schedule and precomputed diffusion constants
- DDPM sampler
- DDIM sampler (50 steps, η=0)
- Classifier-free guidance support (null token at index 10)

---

## Phase 2 — `training.py`

Self-contained script that trains the model and saves `weights.pth`.

It handles:

- CIFAR-10 download and dataloader setup
- U-Net instantiation, optimizer, EMA setup
- Training loop with classifier-free guidance dropout (~15% of steps use null class token)
- Cosine noise schedule
- EMA weight update every step
- Saving the **EMA weights** to `weights.pth` — the checkpoint file the evaluator will point to
- Periodic checkpointing every 10 epochs in case of interruption

No command-line arguments are required for training — hardcode all hyperparameters.

---

## Phase 3 — `inference.py`

Loads weights, generates images, and prints FID and IS scores. Must accept exactly these four arguments:

- `--checkpoint_path` — path to `weights.pth`
- `--label_class` — integer 0–9
- `--number_of_samples` — how many images to generate
- `--output_path` — directory to save generated PNG images

It handles:

- Parsing the four arguments
- Creating the output directory if it does not exist
- Loading the U-Net architecture and restoring weights from `--checkpoint_path`
- Running **both** DDPM and DDIM samplers — decide which one to use by default, or expose a `--sampler` flag as an optional fifth argument with a default value so the evaluator's command still works unchanged
- Generating exactly `--number_of_samples` images for `--label_class`
- Saving each image as a PNG to `--output_path`
- Images saved in uint8 [0, 255] range

### FID and IS Computation (printed after generation)

After all images are saved, the script must automatically compute and print both metrics before exiting. The flow is:

1. **Pull real reference images** — download the CIFAR-10 test split inside the script, filter to only the images matching `--label_class`, and save them to a temporary folder
2. **Compute FID** — use `pytorch-fid` programmatically, comparing the real temp folder against `--output_path`
3. **Compute IS** — use `torch-fidelity` programmatically on the generated images in `--output_path`
4. **Print to stdout** in a clean, readable format, e.g.:

```
Images saved to: /content/your_roll_number/class/images
------------------------------------
FID Score  : 28.43
IS Score   : 6.81 ± 0.12
------------------------------------
```

5. Clean up the temporary real-images folder after metrics are computed

**Important constraint:** The number of generated samples directly affects metric reliability. FID computed on fewer than 1000 images is statistically noisy. Since the evaluator may pass `--number_of_samples 100`, print a warning alongside the scores if the sample count is below 1000, so the evaluator is aware the scores may not be representative. Do not block execution — just warn.

---

## Phase 4 — `modal_training.py`

Root-level file, **not submitted**. Wraps `training.py` for execution on Modal GPU servers.

It handles:

- Defining the Modal container image with all dependencies
- Mounting the `IE624_XXX/` directory into the container
- Configuring GPU (A100 preferred) and timeout
- Running `training.py` remotely
- **Downloading the resulting `weights.pth` back into `IE624_XXX/`** — must be explicit, not optional

---

## Phase 5 — `modal_inference.py`

Root-level file, **not submitted**. Used to generate large image batches for FID/IS evaluation on Modal.

It handles:

- Uploading `weights.pth` into the Modal container
- Running `inference.py` with the correct four arguments
- Downloading generated images back locally for metric computation

---

## Phase 6 — Training Run

1. Run `modal_training.py` to train on Modal
2. Monitor loss — should decrease and stabilize within 150–200 epochs
3. Confirm `weights.pth` is saved back into `IE624_XXX/`
4. Sanity check: run `inference.py` locally with any class label and verify images are visually coherent

---

## Phase 7 — Evaluation

### Setup
- Extract real CIFAR-10 **test split** images for classes X-1, X, X+1 into separate folders
- These are the reference distributions for FID

### Generate Images (6 runs)
Run `inference.py` for each of the three classes using the exact evaluator command format:

```bash
python inference.py --checkpoint_path weights.pth --label_class X --number_of_samples 1000 --output_path ./eval/classX
```

Repeat for X-1 and X+1.

### Compute FID
- Use `pytorch-fid` on each generated folder vs real folder
- Target: FID < 50 acceptable, < 30 strong

### Compute IS
- Use `torch-fidelity` on each generated folder
- Target: IS > 5

### Visual Check
- Inspect a grid of generated images per class — should be recognizable and diverse

---

## Phase 8 — Final Submission

1. Confirm the folder contains exactly three files: `training.py`, `inference.py`, `weights.pth`
2. Run the exact evaluator command on a **clean environment** to verify it works end-to-end
3. Zip the folder: `IE624_XXX.zip`
4. Submit

---

## Key Rules

| Rule | Reason |
|---|---|
| No files other than the three specified | Submission spec is strict |
| Model class defined in both `.py` files | No shared imports between scripts |
| EMA weights saved to `weights.pth` | Raw weights produce visibly worse samples |
| Output dir created if missing in inference | Evaluator should not need to pre-create it |
| Images saved as PNG uint8 | JPEG artifacts corrupt FID statistics |
| Generate 1000+ images for FID eval | FID is statistically unreliable below this |
| FID/IS computed and printed inside `inference.py` | Evaluator sees scores without running separate tools |
| Real reference images fetched from CIFAR-10 test split inside the script | Evaluator does not need to provide a reference dataset |
| Warn if sample count < 1000 when printing scores | Makes evaluator aware scores at 100 samples are indicative only |
| Clean up temporary real-image folder after metrics | No side effects left on evaluator's machine |
