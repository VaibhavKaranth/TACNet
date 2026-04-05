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

| Method   | Accuracy on Reconstructed Images | Notes |
|----------|----------------------------------|-------|
| **TACNet** (β=0.5) | ~64–65% | Task loss guides encoder |
| Baseline (β=0.0) | ~36–38% | Reconstruction only |

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

## Setup

### Requirements

- Python 3.9 or newer
- Windows / Linux / macOS
- No GPU required (CPU training works)

### 1. Clone the repository

```bash
git clone https://github.com/VaibhavKaranth/TACNet.git
cd TACNet
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install gradio matplotlib numpy pillow
```

> For NVIDIA GPU support, get the right torch install command from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Training

Train the full pipeline (classifier + compressor):

```bash
python experiments/run_all.py
```

This runs 4 phases automatically:
1. **Classifier** — trains ResNet-18 on CIFAR-10
2. **TACNet Compressor** — trains with task-aware loss (β=0.5)
3. **Baseline Compressor** — trains without task loss (β=0.0)
4. **Evaluation & Visualization** — saves plots to `results/`

Checkpoints are saved to `checkpoints/`. CIFAR-10 data downloads automatically on first run.

### Adjust hyperparameters

Edit `src/config.py`:

```python
clf_epochs       = 10             # Classifier training epochs
quick_cmp_epochs = 8              # Compressor epochs
gamma_values     = [0.001, 0.01]  # Rate penalty levels to train
cmp_batch_size   = 16             # Reduce if you run out of RAM
```

---

## Running the Demo

### Option A — Windows (easiest)

Double-click **`run_demo.bat`** from File Explorer.  
Then open **http://127.0.0.1:7860** in Chrome or Edge.

### Option B — Command Prompt

```cmd
venv\Scripts\activate
set GRADIO_ANALYTICS_ENABLED=False
set HF_HUB_OFFLINE=1
python app_gradio.py
```

### Option C — Linux / macOS

```bash
source venv/bin/activate
GRADIO_ANALYTICS_ENABLED=False HF_HUB_OFFLINE=1 python app_gradio.py
```

Then open **http://127.0.0.1:7860** in your browser.

### Using the Demo

1. Upload any image (resized to 32×32 internally)
2. Choose a compression level (Low / Medium / High / Very High)
3. Click **Compress & Compare**
4. View results:
   - Side-by-side: Original vs TACNet vs Baseline
   - Metrics table: PSNR, SSIM, BPP, predicted class & confidence
   - Classifier confidence bar chart
   - Error difference maps

---

## Project Structure

```
TACNet/
├── app_gradio.py           Interactive Gradio demo
├── run_demo.bat            Windows one-click demo launcher
├── experiments/
│   └── run_all.py          End-to-end training pipeline (4 phases)
├── src/
│   ├── config.py           All hyperparameters
│   ├── data/dataset.py     CIFAR-10 loaders (auto-download)
│   ├── models/
│   │   ├── classifier.py   ResNet-18 (CIFAR-adapted)
│   │   ├── compressor.py   Encoder / STE Quantizer / Decoder
│   │   └── tacnet.py       Full pipeline
│   ├── losses/rdt_loss.py  Rate–Distortion–Task loss
│   ├── train/              Classifier + compressor training loops
│   ├── evaluate/metrics.py PSNR · SSIM · BPP · Accuracy
│   └── utils/              Device detection · Visualization
└── checkpoints/            Saved model weights (after training)
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `python experiments/run_all.py` | Full training pipeline |
| `python app_gradio.py` | Launch interactive demo |

---

## Windows Notes

- `num_workers=0` is set in all DataLoaders — required to avoid multiprocessing crashes on Windows
- Run with `set PYTHONUTF8=1` if you see Unicode/encoding errors in the terminal
- `run_demo.bat` handles all environment variables automatically

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
