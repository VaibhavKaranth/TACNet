# TACNet: Task-Aware Image Compression Optimized for Classification Accuracy

## Vision

TACNet addresses a fundamental disconnect in modern AI pipelines: images compressed for
human viewers are increasingly decoded by machine learning models, not human eyes. Standard
codecs (JPEG, WebP) and even learned compression methods optimize for perceptual quality —
metrics like PSNR and SSIM — that measure how faithfully pixels are reconstructed. But
machines don't see pixels; they see features. When you strip edge orientations and semantic
textures to save bits, you may save bandwidth while destroying classification accuracy.

**TACNet's vision: compress images not to look good, but to be understood.**

---

## Problem Statement

Traditional image compression optimizes for human-visual reconstruction quality (PSNR/SSIM).
In modern AI pipelines, images are consumed by classification and recognition models — not
human observers. Pixel-optimized compression removes features crucial for recognition.

TACNet proposes a **task-aware neural image compression framework** that compresses images
while explicitly preserving classification accuracy at low bitrates.

---

## Research Gap

- Most compression methods (JPEG, learned) focus exclusively on pixel fidelity metrics.
- Few student-level implementations clearly demonstrate the rate–accuracy tradeoff.
- Practical need exists for compression that preserves **machine perception**, not only human
  perception.
- Existing task-aware methods (AccelIR, Tombo) are complex; a clean, reproducible baseline is
  absent in the student implementation space.

---

## Novelty

1. **Rate–Distortion–Task (RDT) Objective**: Joint optimization of three competing goals:
   - Distortion (pixel reconstruction fidelity)
   - Bitrate (compression ratio)
   - Task performance (classification accuracy)

2. **Accuracy vs BPP Curves**: Systematic sweep across bitrate operating points, demonstrating
   that TACNet maintains higher classification accuracy than reconstruction-only baselines at
   equivalent bitrates.

3. **Task-Guided Bottleneck**: The frozen classifier's gradient signal guides what semantic
   information the encoder retains during compression, without any additional model complexity.

---

## Technical Architecture

### System Overview

```
Input Image x  [B, 3, 32, 32]  in [0, 1]
      │
      ▼
 ┌─────────────────────────────────┐
 │           ENCODER  E(x)         │  CNN downsampling
 │  3×32×32 → 64×32×32 → 128×16×16│
 │         → 256×8×8 → C×8×8      │
 └─────────────────────────────────┘
      │  z  [B, C, 8, 8]
      ▼
 ┌─────────────────────────────────┐
 │        QUANTIZER  Q(z)          │  Straight-Through Estimator
 │  z_hat = round(z)               │  (differentiable in backprop)
 └─────────────────────────────────┘
      │  z_hat  [B, C, 8, 8]
      ▼
 ┌─────────────────────────────────┐
 │           DECODER  G(z_hat)     │  CNN upsampling
 │  C×8×8 → 128×16×16 → 64×32×32  │
 │         → 3×32×32               │
 └─────────────────────────────────┘
      │  x_hat  [B, 3, 32, 32]  in [0, 1]
      │
      ├──── L_rec = MSE(x_hat, x)
      │
      └──── normalize(x_hat) ──► ┌───────────────────┐
                                  │   CLASSIFIER  C   │  ResNet-18 [FROZEN]
                                  │   (pre-trained on  │
                                  │    clean CIFAR-10) │
                                  └───────────────────┘
                                         │
                                  L_task = CE(C(x_hat_norm), y)

Rate penalty:  L_rate = mean(|z|)

Joint Loss:    L = α·L_rec + β·L_task + γ·L_rate
```

### Architecture Details

#### Encoder
CNN with progressive spatial downsampling:

| Layer | Operation | Output Shape |
|-------|-----------|-------------|
| 0 | Conv2d(3→64, k=3, s=1, p=1) + BN + LeakyReLU | 64×32×32 |
| 1 | Conv2d(64→128, k=3, s=2, p=1) + BN + LeakyReLU | 128×16×16 |
| 2 | Conv2d(128→256, k=3, s=2, p=1) + BN + LeakyReLU | 256×8×8 |
| 3 | Conv2d(256→C, k=3, s=1, p=1) | C×8×8 |

#### Decoder
CNN with progressive spatial upsampling:

| Layer | Operation | Output Shape |
|-------|-----------|-------------|
| 0 | Conv2d(C→256, k=3, s=1, p=1) + BN + LeakyReLU | 256×8×8 |
| 1 | ConvTranspose2d(256→128, k=4, s=2, p=1) + BN + LeakyReLU | 128×16×16 |
| 2 | ConvTranspose2d(128→64, k=4, s=2, p=1) + BN + LeakyReLU | 64×32×32 |
| 3 | Conv2d(64→3, k=3, s=1, p=1) + Sigmoid | 3×32×32 |

#### Quantizer (Straight-Through Estimator)
- **Forward**: `z_hat = round(z)` — discrete, non-differentiable
- **Backward**: gradient treated as identity — `∂z_hat/∂z = 1`
- Implementation: `z_hat = z + (round(z) - z).detach()`

#### Frozen Classifier (ResNet-18 on CIFAR-10)
- Adapted: first conv is 3×3 stride-1 (not 7×7 stride-2); no max-pool
- Trained on clean, normalized CIFAR-10 to ≥90% test accuracy
- All parameters frozen during compressor training
- Receives **normalized** reconstructed images: `normalize(x_hat)`

---

## Loss Function

### Rate–Distortion–Task (RDT) Loss

```
L = α · L_rec + β · L_task + γ · L_rate
```

| Term | Formula | Role |
|------|---------|------|
| `L_rec` | `MSE(x_hat, x)` | Pixel reconstruction fidelity |
| `L_task` | `CrossEntropy(C(normalize(x_hat)), y)` | Classification preservation |
| `L_rate` | `mean(|z|)` | Bitrate proxy (penalizes large latents) |
| `α` | 1.0 (default) | Reconstruction weight |
| `β` | 0.5 (TACNet) / 0.0 (baseline) | Task weight |
| `γ` | varied ∈ [0.0001 … 0.1] | Rate weight (controls BPP) |

**Baseline**: set `β = 0` → reconstruction-only compression (no task awareness)

---

## Dataset

### Primary: CIFAR-10
- **Source**: Krizhevsky (2009); auto-downloaded via `torchvision.datasets.CIFAR10`
- **Size**: 60,000 color images, 32×32 pixels, 10 classes
- **Split**: 45,000 train / 5,000 val / 10,000 test
- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Preprocessing**:
  - Classifier training: RandomCrop(32, pad=4) + RandomHorizontalFlip + Normalize(μ,σ)
  - Compressor training/evaluation: ToTensor() → [0, 1] range (no normalization)
  - Classifier receives **normalized** images; compressor works in **[0,1]** space

### CIFAR-10 Normalization Statistics
```
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)
```

---

## Training Strategy

### Phase 1: Classifier Pre-training
1. Load CIFAR-10 with standard augmentation
2. Train ResNet-18 (CIFAR-adapted) for 50 epochs
3. Optimizer: SGD, lr=0.1, momentum=0.9, weight_decay=5e-4
4. LR Schedule: CosineAnnealingLR
5. Target: ≥90% test accuracy
6. Save best checkpoint; freeze all parameters

### Phase 2: Multi-Rate Compressor Training
1. Load frozen classifier from checkpoint
2. Train compressor (encoder + decoder) with joint RDT loss
3. Optimizer: Adam, lr=1e-3
4. LR Schedule: CosineAnnealingLR over 100 epochs
5. Vary `γ ∈ [0.0001, 0.001, 0.01, 0.1]` → 4 operating points per method
6. Train **TACNet** (β=0.5) and **Baseline** (β=0.0) at each γ → 8 models total

### Phase 3: Evaluation
- Evaluate all 8 models on CIFAR-10 test set
- Report Accuracy, BPP, PSNR, SSIM per model

---

## Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Accuracy** | top-1 on reconstructed images | Primary task metric |
| **BPP** | empirical entropy of z_hat / pixels | Bits per pixel (compression ratio) |
| **PSNR** | 10·log₁₀(1/MSE) dB | Reconstruction quality |
| **SSIM** | structural similarity index | Perceptual quality |

### BPP Estimation
```
BPP_empirical = (n_symbols × H(z_hat)) / (image_H × image_W)

where:
  n_symbols = C × H_z × W_z  (symbols per image)
  H(z_hat)  = -Σ p_i · log2(p_i)  (empirical entropy of quantized values)
```
Theoretical upper bound: `BPP_max = C × H_z × W_z × 8 / (32 × 32) = C × 0.5`

---

## Expected Results

### Quantitative (Table: 3 bitrate levels × 2 methods)

| Method | BPP | Accuracy (%) | PSNR (dB) | SSIM |
|--------|-----|-------------|-----------|------|
| TACNet (low rate) | ~1.0 | **higher** | slightly lower | similar |
| Baseline (low rate) | ~1.0 | lower | slightly higher | similar |
| TACNet (med rate) | ~2.0 | **higher** | slightly lower | similar |
| Baseline (med rate) | ~2.0 | lower | slightly higher | similar |
| TACNet (high rate) | ~3.5 | **≈similar** | ≈similar | ≈similar |
| Baseline (high rate) | ~3.5 | ≈similar | ≈similar | ≈similar |

**Key publishable result**: At the same BPP, TACNet achieves higher classification accuracy.

### Qualitative
- At low bitrate: TACNet preserves class-discriminative features (textures, shapes)
- Baseline at same bitrate: blurry, loses fine-grained discriminative details

### Ablation
- Removing task loss (β=0) → accuracy drops at low BPP
- Confirms task loss is the key differentiating component

---

## Device Strategy

```
GPU (CUDA)  →  Full training (fast, recommended)
Apple MPS   →  Full training (moderate speed)
CPU only    →  Full training (slow; consider --quick flag for reduced epochs)
```

Detection is automatic. No code changes needed.

---

## Project Structure

```
TACNet/
├── vision.md                      # This document (written first)
├── requirements.txt               # Python dependencies
├── main.py                        # CLI entry point
│
├── src/
│   ├── config.py                  # All hyperparameters and paths
│   ├── data/
│   │   └── dataset.py             # CIFAR-10 loaders (normalized + raw)
│   ├── models/
│   │   ├── classifier.py          # ResNet-18 adapted for CIFAR-10
│   │   ├── compressor.py          # Encoder / STE Quantizer / Decoder
│   │   └── tacnet.py              # Full TACNet pipeline
│   ├── losses/
│   │   └── rdt_loss.py            # Rate-Distortion-Task loss
│   ├── train/
│   │   ├── train_classifier.py    # Phase 1: classifier training
│   │   └── train_tacnet.py        # Phase 2: compressor training
│   ├── evaluate/
│   │   └── metrics.py             # BPP, PSNR, SSIM, accuracy
│   └── utils/
│       ├── device.py              # Auto GPU/CPU detection
│       └── visualization.py       # Plots and qualitative grids
│
├── experiments/
│   └── run_all.py                 # Full end-to-end pipeline
│
├── data/                          # Auto-downloaded CIFAR-10
├── checkpoints/                   # Saved model weights
└── results/                       # Plots, tables, JSON results
```

---

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (recommended)
python main.py --mode run_all

# Train only classifier
python main.py --mode train_classifier

# Train one TACNet model
python main.py --mode train_tacnet --gamma 0.01 --beta 0.5

# Train one baseline model (no task loss)
python main.py --mode train_tacnet --gamma 0.01 --beta 0.0 --name baseline_g0_01

# Evaluate a trained model
python main.py --mode evaluate --name tacnet_gamma0_0100_beta0_50

# Quick mode for CPU (reduced epochs)
python main.py --mode run_all --quick
```

---

## References

**Foundational**
1. J. Ballé et al., "Variational image compression with a scale hyperprior," IEEE TPAMI, 2020.
2. J. Ballé et al., "End-to-end optimized image compression," ICLR, 2017.
3. L. Theis et al., "Lossy image compression with compressive autoencoders," ICLR, 2017.
4. D. Minnen et al., "Joint autoregressive and hierarchical priors for learned image compression," NeurIPS, 2018.

**Task-Aware Compression**
5. J. Ye et al., "AccelIR: Task-Aware Image Compression for Accelerating Neural Image Restoration," CVPR, 2023.
6. H. Li et al., "Image Compression for Machine and Human Vision with Adaptation," ECCV, 2024.
7. S. Yan et al., "Tombo: Task-Oriented Multi-Bitstream Optimization," ACM, 2024.
8. "Progressive Learned Image Compression for Machine Perception," arXiv, 2025.
9. D. He et al., "ELIC: Efficient Learned Image Compression," CVPR, 2022.

**Backbone & Metrics**
10. K. He et al., "Deep residual learning for image recognition," CVPR, 2016.
11. A. Krizhevsky, "Learning multiple layers of features from tiny images," Tech. Rep., 2009.
12. Z. Wang et al., "Image quality assessment: SSIM," IEEE TIP, 2004.
