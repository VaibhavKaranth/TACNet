# TACNet — Task-Aware Image Compression

> Compress images **to be understood**, not just to look good.

TACNet is a neural image compression framework that jointly optimises reconstruction quality, bitrate, and classification accuracy using a **Rate–Distortion–Task (RDT)** loss.

---

## The Idea

Standard compression (JPEG, learned codecs) minimises pixel error.
When a classifier sees the result, it may fail — the features it needs are gone.
TACNet keeps a **frozen ResNet-18** in the training loop so the encoder learns to preserve what matters for recognition.

```
L = α · MSE(x̂, x)  +  β · CE(Classifier(x̂), y)  +  γ · mean(|z|)
      reconstruction        task accuracy                 bitrate
```

---

## Results

At equal BPP, TACNet yields **higher classification accuracy** than reconstruction-only baselines.
A small PSNR drop is acceptable — the rate–accuracy curve is the key publishable result.

| Method   | BPP  | Accuracy | PSNR  | SSIM  |
|----------|------|----------|-------|-------|
| TACNet   | ~1.5 | **higher** | slightly lower | similar |
| Baseline | ~1.5 | lower    | slightly higher | similar |

---

## Architecture

```
x [3×32×32]
  → Encoder (CNN, ×2 stride-2)
  → z [C×8×8]
  → STE Quantizer  (round + straight-through gradient)
  → z_hat
  → Decoder (CNN, ×2 transposed)
  → x̂ [3×32×32]
  → frozen ResNet-18 → logits → L_task
```

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Train (GPU auto-detected; use --quick on CPU)
python main.py --mode run_all
python main.py --mode run_all --quick   # reduced epochs for CPU

# 3. Launch interactive demo
python app_gradio.py                    # → http://localhost:7860
```

---

## Project Structure

```
TACNet/
├── main.py                 CLI entry point
├── app_gradio.py           Interactive Gradio demo
├── vision.md               Full project vision & architecture
├── src/
│   ├── config.py           All hyperparameters
│   ├── data/dataset.py     CIFAR-10 loaders (auto-download)
│   ├── models/
│   │   ├── classifier.py   ResNet-18 (CIFAR-adapted)
│   │   ├── compressor.py   Encoder / STE Quantizer / Decoder
│   │   └── tacnet.py       Full pipeline
│   ├── losses/rdt_loss.py  Rate–Distortion–Task loss
│   ├── train/              Classifier + compressor training
│   ├── evaluate/metrics.py PSNR · SSIM · BPP · Accuracy
│   └── utils/              Device detection · Visualization
└── experiments/run_all.py  End-to-end pipeline (4 phases)
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `python main.py --mode run_all` | Full pipeline |
| `python main.py --mode run_all --quick` | Fast CPU run |
| `python main.py --mode train_classifier` | Phase 1 only |
| `python main.py --mode train_tacnet --gamma 0.01 --beta 0.5` | One TACNet model |
| `python main.py --mode train_tacnet --gamma 0.01 --beta 0.0` | One baseline model |
| `python main.py --mode evaluate --name tacnet_gamma0_0100` | Evaluate saved model |
| `python app_gradio.py` | Launch demo UI |

---

## Dataset

CIFAR-10 — downloaded automatically on first run via `torchvision`.
50,000 train · 10,000 test · 10 classes · 32×32 pixels.

---

## Key References

- Ballé et al., *End-to-end optimized image compression*, ICLR 2017
- Ballé et al., *Variational image compression with a scale hyperprior*, IEEE TPAMI 2020
- He et al., *Deep residual learning for image recognition*, CVPR 2016
- Ye et al., *AccelIR: Task-Aware Image Compression*, CVPR 2023
- He et al., *ELIC: Efficient Learned Image Compression*, CVPR 2022
