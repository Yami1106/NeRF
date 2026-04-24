# NeRF: Neural Radiance Fields — From Scratch

A from-scratch PyTorch implementation of [NeRF (Mildenhall et al., 2020)](https://arxiv.org/abs/2003.08934) for novel view synthesis from 2D images with known camera poses. This project was developed as part of WPI's computer vision coursework.

---

## Project Overview

NeRF reconstructs a continuous 3D scene representation by training a neural network (MLP) to map a 5D input — a 3D spatial position **(x, y, z)** and a 2D viewing direction **(θ, φ)** — to an output color **(r, g, b)** and volume density **σ**. Once trained, the network can synthesize photorealistic novel views of the scene from any camera angle.

Key components implemented:
- **Positional Encoding** — lifts raw coordinates into high-frequency sinusoidal feature vectors to help the MLP learn fine detail
- **Hierarchical Sampling** — a coarse network proposes sample distributions along each ray; a fine network re-samples in high-density regions
- **Volume Rendering** — accumulates color and opacity along each ray using the classical rendering integral
- **White Background Compositing** — fills unoccupied ray fractions with a white background, matching the synthetic dataset

---

## File Structure

```
.
├── Wrapper.py       # Dataset loading, ray generation, rendering, training & testing pipeline
├── NeRFModel.py     # 8-layer MLP architecture with positional encoding
├── data/
│   └── <scene>/     # NeRF synthetic dataset (e.g. lego/, ship/)
│       ├── transforms_train.json
│       ├── transforms_test.json
│       ├── transforms_val.json
│       └── train/ test/ val/  (PNG images)
├── logs/            # TensorBoard training logs (auto-created)
├── checkpoints/     # Saved model checkpoints (auto-created)
└── images/          # Rendered output images & GIF (auto-created)
```

---

## Requirements

Install dependencies with:

```bash
pip install torch torchvision numpy imageio scikit-image matplotlib tqdm tensorboard
```

> **GPU strongly recommended.** Training on CPU will be very slow.  
> Tested with Python 3.8+, PyTorch 1.12+.

---

## Dataset

This implementation uses the **NeRF Blender Synthetic Dataset**. Download it from the [original NeRF repository](https://github.com/bmild/nerf) or [this mirror](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).

Scenes available: `chair`, `drums`, `ficus`, `hotdog`, `lego`, `materials`, `mic`, `ship`.

Each scene contains:
- 100 training images, 100 validation images, 200 test images at 800×800 resolution
- JSON files with camera poses (4×4 transform matrices) and field-of-view

---

## Training

```bash
python Wrapper.py \
  --mode train \
  --data_path ./data/lego/ \
  --checkpoint_path ./checkpoints/lego/ \
  --images_path ./images/lego/ \
  --logs_path ./logs/lego/ \
  --max_iters 100000 \
  --n_rays_batch 4096 \
  --n_sample 64 \
  --n_sample_fine 128 \
  --n_pos_freq 10 \
  --n_dirc_freq 4 \
  --lrate 5e-4 \
  --save_ckpt_iter 1000 \
  --near 2.0 \
  --far 6.0
```

Checkpoints are saved every `--save_ckpt_iter` iterations. Training can be resumed automatically from the latest checkpoint if `--load_checkpoint True` (default).

Monitor training loss live with TensorBoard:
```bash
tensorboard --logdir ./logs/lego/
```

---

## Testing / Rendering

After training, run inference to render all test views:

```bash
python Wrapper.py \
  --mode test \
  --data_path ./data/lego/ \
  --checkpoint_path ./checkpoints/lego/ \
  --images_path ./images/lego/ \
  --chunk_size 4096 \
  --n_pos_freq 10 \
  --n_dirc_freq 4 \
  --n_sample 64 \
  --n_sample_fine 128 \
  --near 2.0 \
  --far 6.0
```

This will:
1. Load the latest checkpoint from `--checkpoint_path`
2. Render all test images and save them to `--images_path`
3. Generate a `360_view.gif` from the rendered frames
4. Compute and print **PSNR** and **SSIM** metrics
5. Save best/worst comparison images side-by-side with ground truth

---

## All CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_path` | `./Phase2/data/lego/` | Path to dataset directory |
| `--mode` | `train` | `train` or `test` |
| `--lrate` | `5e-4` | Adam optimizer learning rate |
| `--n_pos_freq` | `10` | Positional encoding frequencies for (x,y,z) |
| `--n_dirc_freq` | `4` | Positional encoding frequencies for direction |
| `--n_rays_batch` | `4096` | Rays sampled per training iteration |
| `--n_sample` | `64` | Coarse samples per ray |
| `--n_sample_fine` | `128` | Fine (hierarchical) samples per ray |
| `--max_iters` | `10000` | Total training iterations |
| `--near` | `2.0` | Near clipping distance for ray sampling |
| `--far` | `6.0` | Far clipping distance for ray sampling |
| `--logs_path` | `./logs/` | TensorBoard log directory |
| `--checkpoint_path` | `./Phase2/example_checkpoint/` | Checkpoint save/load directory |
| `--load_checkpoint` | `True` | Resume from latest checkpoint if available |
| `--save_ckpt_iter` | `1000` | Save checkpoint every N iterations |
| `--images_path` | `./image/` | Output image directory |
| `--chunk_size` | `4096` | Rays rendered per chunk during testing (controls VRAM usage) |

---

## Results

| Dataset | Iterations | Final Loss | PSNR (dB) | SSIM |
|---|---|---|---|---|
| Lego | 100,000 | 0.004216 | 27.42 | 0.9084 |
| Ship | 70,000 | 0.005386 | 25.75 | 0.7991 |

---

## Architecture Notes (`NeRFModel.py`)

The MLP consists of 8 fully-connected layers (256 hidden units each) with a skip connection at layer 5 that re-injects the positional encoding. Volume density **σ** is predicted after layer 8. Color **RGB** is predicted by a separate branch that receives both the layer-8 feature vector and the direction encoding, allowing view-dependent appearance effects (specularities, reflections).

```
pos_enc (63-dim) ──► FC1─FC2─FC3─FC4 ──┐
                                         ├─ cat ──► FC5─FC6─FC7─FC8 ──► σ
pos_enc (63-dim) ──────────────────────┘                        │
                                                                 └──► feature
dir_enc (27-dim) ─────────────────────────────────────────────── cat ──► FC_dir ──► RGB
```

---

## Known Issues & Fixes

- **Black pixels early in training**: Fixed by adding a white background compositing term — `C(r) = Σ wᵢcᵢ + (1 − Σ wᵢ) × white`
- **Without positional encoding**: ReLU dead-neuron collapse causes the network to output a plain white image. Fixed by switching density activation to **Softplus**, which provides a small positive gradient even for negative pre-activations

---

## References

- Mildenhall et al., *"NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"*, arXiv:2003.08934, 2020