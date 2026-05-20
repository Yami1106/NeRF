<div align="center">

# NeRF — Neural Radiance Fields from Scratch

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)

*Teaching a neural network to understand 3D space from 2D photos — achieving 27.42 dB PSNR on synthetic scenes and trained on real-world captures.*

</div>

---

## What is NeRF?

Neural Radiance Fields represent a 3D scene as a continuous function learned by an MLP. Given any camera position and direction, the network predicts the colour and density at every point along the ray — enabling photorealistic **novel view synthesis** from a sparse set of images.

---

## Implementation

Every component built from scratch:

```
Camera rays → Positional encoding → Coarse MLP (uniform sampling)
           → Importance sampling → Fine MLP → Volume rendering → RGB image
```

**Volume rendering equation:**
```
C(r) = Σ Tᵢ (1 - exp(-σᵢδᵢ)) cᵢ    where    Tᵢ = exp(-Σⱼ<ᵢ σⱼδⱼ)
```

### Architecture
- **Input:** 5D — position (x, y, z) + viewing direction (θ, φ)
- **Encoding:** sinusoidal positional encoding at multiple frequencies
- **Network:** coarse + fine MLP with hierarchical sampling
- **Output:** RGB colour + volume density σ

---

## Results

| Dataset | Iterations | PSNR | SSIM |
|---|---|---|---|
| Lego (synthetic) | 100k | **27.42 dB** | 0.9084 |
| Ship (synthetic) | 70k | 25.75 dB | 0.7991 |
| Custom (78 imgs, Samsung S23) | — | 20.13 dB | — |
| Custom (341 imgs) | — | 19.65 dB | — |

---

## Ablation findings

| Change | Effect |
|---|---|
| Remove positional encoding | Output collapses to pure white — MLP cannot learn high-frequency detail |
| ReLU → Softplus activation | Fixed dead neurons during early training |
| Background correction | Accuracy map + white fill for unabsorbed rays |

---

## Tech stack

`Python` · `PyTorch` · `NumPy` · `COLMAP` (for custom dataset camera poses)

---

<div align="center">
Part of the WPI Computer Vision course · <a href="https://github.com/Yami1106">Ashish Sukumar</a>
</div>
