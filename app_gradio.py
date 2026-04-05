"""
TACNet — Gradio Interactive Demo
=================================
Upload any image → choose compression strength → compare TACNet vs Baseline.

Run:
    python app_gradio.py

Then open:  http://localhost:7860
"""

import os
import sys
import io

# Disable Gradio analytics and update checks (prevents network hang on import)
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["GRADIO_SERVER_NAME"] = "127.0.0.1"

import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running from project root
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import Config
from src.utils.device import get_device
from src.models.tacnet import TACNet
from src.evaluate.metrics import compute_psnr, compute_ssim, compute_bpp
from src.data.dataset import CIFAR10_CLASSES

# ── Startup ───────────────────────────────────────────────────────────────────

config = Config()
device = get_device()

# Compression levels → γ values (low γ = low compression / high quality)
GAMMA_VALUES = [0.0001, 0.001, 0.01, 0.1]
LEVEL_NAMES  = [
    "Low Compression    (γ=0.0001)  — highest quality",
    "Medium Compression (γ=0.001)",
    "High Compression   (γ=0.01)",
    "Very High Compression (γ=0.1) — lowest quality",
]

# ── Model loading ─────────────────────────────────────────────────────────────

tacnet_models   : dict = {}
baseline_models : dict = {}
classifier_path = os.path.join(config.checkpoints_dir, "classifier_best.pth")


def _load_all_models() -> str:
    """Load all available checkpoints. Returns status string."""
    loaded_t = []
    loaded_b = []

    if not os.path.exists(classifier_path):
        return "no_classifier"

    for gamma in GAMMA_VALUES:
        g_tag = f"{gamma:.4f}".replace(".", "_")

        t_path = os.path.join(config.checkpoints_dir, f"tacnet_gamma{g_tag}.pth")
        b_path = os.path.join(config.checkpoints_dir, f"baseline_gamma{g_tag}.pth")

        if os.path.exists(t_path):
            m = TACNet(config.cmp_latent_channels, config.num_classes).to(device)
            m.load_classifier(classifier_path, device)
            data = torch.load(t_path, map_location=device)
            m.compressor.load_state_dict(data["compressor_state_dict"])
            m.eval_mode()
            tacnet_models[gamma] = m
            loaded_t.append(gamma)

        if os.path.exists(b_path):
            m = TACNet(config.cmp_latent_channels, config.num_classes).to(device)
            m.load_classifier(classifier_path, device)
            data = torch.load(b_path, map_location=device)
            m.compressor.load_state_dict(data["compressor_state_dict"])
            m.eval_mode()
            baseline_models[gamma] = m
            loaded_b.append(gamma)

    if loaded_t:
        return f"ok:{len(loaded_t)}"
    return "no_models"


_status = _load_all_models()


def _status_message() -> str:
    if _status == "no_classifier":
        return (
            "⚠️  **Classifier not trained yet.**\n\n"
            "Run: `python main.py --mode train_classifier`"
        )
    if _status == "no_models":
        return (
            "⚠️  **No compression models found in `./checkpoints/`.**\n\n"
            "Run: `python main.py --mode run_all` to train everything first.\n\n"
            "Quick CPU run: `python main.py --mode run_all --quick`"
        )
    n = int(_status.split(":")[1])
    from src.utils.device import device_summary
    return (
        f"✅  **{n} TACNet + {n} Baseline model(s) loaded** "
        f"| Device: {device_summary()}"
    )


# ── Image utilities ───────────────────────────────────────────────────────────

def _preprocess(pil_img: Image.Image) -> torch.Tensor:
    """PIL image (any size) → [1, 3, 32, 32] float tensor in [0, 1]."""
    img   = pil_img.convert("RGB").resize((32, 32), Image.LANCZOS)
    arr   = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def _to_display(tensor: torch.Tensor, scale: int = 8) -> Image.Image:
    """[1, 3, H, W] tensor in [0,1] → upscaled PIL image for display."""
    arr = tensor.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    W, H = pil.size
    # NEAREST = preserve pixel art look at small resolution
    return pil.resize((W * scale, H * scale), Image.NEAREST)


# ── Probability bar chart ─────────────────────────────────────────────────────

def _prob_chart(logits_t: torch.Tensor, logits_b: torch.Tensor, gamma: float) -> Image.Image:
    """Grouped bar chart comparing class probabilities from both models."""
    probs_t = F.softmax(logits_t.squeeze(0), dim=-1).cpu().numpy()
    probs_b = F.softmax(logits_b.squeeze(0), dim=-1).cpu().numpy()

    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(CIFAR10_CLASSES))
    w = 0.38

    ax.bar(x - w / 2, probs_t, w, label="TACNet",   color="#2563EB", alpha=0.85)
    ax.bar(x + w / 2, probs_b, w, label="Baseline",  color="#DC2626", alpha=0.85)

    # Highlight top prediction for each model
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.set_title(
        f"Classifier Confidence on Reconstructed Images  (γ={gamma})",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return Image.open(buf).copy()


# ── Difference map ────────────────────────────────────────────────────────────

def _diff_map(orig: torch.Tensor, compressed: torch.Tensor, label: str) -> Image.Image:
    """Amplified absolute difference map (×10 for visibility)."""
    diff = (orig - compressed).abs().clamp(0, 1)
    diff_amplified = (diff * 10).clamp(0, 1)

    fig, ax = plt.subplots(figsize=(3, 3))
    arr = diff_amplified.squeeze(0).permute(1, 2, 0).cpu().numpy()
    im  = ax.imshow(arr, cmap="hot", interpolation="nearest")
    ax.set_title(f"Error map\n{label}", fontsize=9, fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(pad=0.2)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    plt.close()
    buf.seek(0)
    return Image.open(buf).copy()


# ── Main inference ────────────────────────────────────────────────────────────

@torch.no_grad()
def run_demo(pil_image, level_name):
    """Core demo function called by Gradio on button click."""

    # Guard: no image
    if pil_image is None:
        empty = None
        return empty, empty, empty, "Please upload an image.", empty, empty, empty

    # Guard: models not loaded
    if not tacnet_models:
        msg = _status_message()
        return None, None, None, msg, None, None, None

    # Resolve gamma
    level_idx = LEVEL_NAMES.index(level_name)
    gamma     = GAMMA_VALUES[level_idx]

    if gamma not in tacnet_models:
        msg = (
            f"⚠️  No model for γ={gamma}. "
            "Run training for this γ level first."
        )
        return None, None, None, msg, None, None, None

    # ── Inference ─────────────────────────────────────────────────────────────
    x = _preprocess(pil_image)   # [1, 3, 32, 32]

    x_hat_t, z_t, z_hat_t, logits_t = tacnet_models[gamma](x)
    x_hat_b, z_b, z_hat_b, logits_b = baseline_models.get(gamma, tacnet_models[gamma])(x)

    # ── Images ────────────────────────────────────────────────────────────────
    disp_scale  = 8
    orig_disp   = _to_display(x,       disp_scale)
    tacnet_disp = _to_display(x_hat_t, disp_scale)
    base_disp   = _to_display(x_hat_b, disp_scale)

    # ── Metrics ───────────────────────────────────────────────────────────────
    psnr_t = compute_psnr(x, x_hat_t)
    psnr_b = compute_psnr(x, x_hat_b)
    ssim_t = compute_ssim(x, x_hat_t)
    ssim_b = compute_ssim(x, x_hat_b)
    bpp_t  = compute_bpp(z_hat_t, 32, 32)
    bpp_b  = compute_bpp(z_hat_b, 32, 32)

    probs_t   = F.softmax(logits_t.squeeze(0), dim=-1)
    probs_b   = F.softmax(logits_b.squeeze(0), dim=-1)
    pred_t    = CIFAR10_CLASSES[probs_t.argmax().item()]
    conf_t    = probs_t.max().item() * 100
    pred_b    = CIFAR10_CLASSES[probs_b.argmax().item()]
    conf_b    = probs_b.max().item() * 100

    # Winner indicators
    acc_win  = "🟦 TACNet" if conf_t >= conf_b else "🟥 Baseline"
    psnr_win = "🟦 TACNet" if psnr_t >= psnr_b else "🟥 Baseline"
    bpp_win  = "🟦 TACNet" if bpp_t  <= bpp_b  else "🟥 Baseline"

    metrics_md = f"""
### Compression Results  (γ = {gamma})

| Metric | 🟦 TACNet | 🟥 Baseline | Winner |
|--------|-----------|------------|--------|
| **BPP** (↓ better) | {bpp_t:.3f} | {bpp_b:.3f} | {bpp_win} |
| **PSNR** dB (↑ better) | {psnr_t:.2f} | {psnr_b:.2f} | {psnr_win} |
| **SSIM** (↑ better) | {ssim_t:.4f} | {ssim_b:.4f} | {"🟦 TACNet" if ssim_t >= ssim_b else "🟥 Baseline"} |
| **Predicted class** | **{pred_t}** ({conf_t:.1f}%) | **{pred_b}** ({conf_b:.1f}%) | {acc_win} |

> *TACNet trades a small PSNR drop for higher classification confidence — the key result.*
"""

    # ── Charts ────────────────────────────────────────────────────────────────
    prob_chart = _prob_chart(logits_t, logits_b, gamma)
    diff_t     = _diff_map(x, x_hat_t, "TACNet")
    diff_b     = _diff_map(x, x_hat_b, "Baseline")

    return (
        orig_disp,
        tacnet_disp,
        base_disp,
        metrics_md.strip(),
        prob_chart,
        diff_t,
        diff_b,
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
#app-title { text-align: center; }
#status-bar { background: #f0f9ff; border-left: 4px solid #2563eb;
              padding: 0.6rem 1rem; border-radius: 4px; margin-bottom: 0.5rem; }
.metric-table { font-size: 0.95rem; }
"""

with gr.Blocks(
    title="TACNet — Task-Aware Image Compression",
) as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.Markdown(
        "# 🗜️ TACNet: Task-Aware Image Compression\n"
        "**Compress for machine perception, not just human eyes.**  "
        "TACNet optimises a joint _Rate–Distortion–Task_ loss to preserve "
        "classification accuracy at low bitrates.",
        elem_id="app-title",
    )
    gr.Markdown(_status_message(), elem_id="status-bar")

    # ── Controls ──────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            inp_image = gr.Image(
                type="pil",
                label="Upload Image  (any size — resized to 32×32 internally)",
            )
            inp_level = gr.Dropdown(
                choices=LEVEL_NAMES,
                value=LEVEL_NAMES[1],
                label="Compression Level",
                info="Higher compression → lower BPP → harder for classifiers",
            )
            btn_run = gr.Button("⚡ Compress & Compare", variant="primary", size="lg")

        # ── Image outputs ──────────────────────────────────────────────────────
        with gr.Column(scale=3):
            with gr.Row():
                out_orig     = gr.Image(label="📷 Original (32×32 × 8)", height=256)
                out_tacnet   = gr.Image(label="🟦 TACNet Compressed",    height=256)
                out_baseline = gr.Image(label="🟥 Baseline Compressed",  height=256)

    # ── Metrics ───────────────────────────────────────────────────────────────
    out_metrics = gr.Markdown(elem_classes=["metric-table"])

    # ── Charts ────────────────────────────────────────────────────────────────
    with gr.Row():
        out_prob_chart = gr.Image(
            label="📊 Classifier Confidence on Reconstructed Images",
        )

    with gr.Row():
        out_diff_tacnet   = gr.Image(label="🗺️ Error Map — TACNet",   height=200)
        out_diff_baseline = gr.Image(label="🗺️ Error Map — Baseline", height=200)

    # ── Wire up ───────────────────────────────────────────────────────────────
    btn_run.click(
        fn=run_demo,
        inputs=[inp_image, inp_level],
        outputs=[
            out_orig,
            out_tacnet,
            out_baseline,
            out_metrics,
            out_prob_chart,
            out_diff_tacnet,
            out_diff_baseline,
        ],
    )

    # ── Info accordion ────────────────────────────────────────────────────────
    with gr.Accordion("ℹ️  How it works", open=False):
        gr.Markdown("""
**Pipeline**
1. Your image is resized to **32×32** (CIFAR-10 resolution) and converted to `[0,1]` range.
2. The **Encoder** (CNN) maps it to a compact latent code `z` (shape `C×8×8`).
3. The **STE Quantizer** rounds `z` to integers — differentiable via straight-through gradient.
4. The **Decoder** reconstructs `x̂` from the quantised latent.
5. The **Frozen ResNet-18 Classifier** predicts the class from `normalize(x̂)`.

**Loss Function**
```
L = α · MSE(x̂, x)  +  β · CrossEntropy(C(x̂), y)  +  γ · mean(|z|)
      reconstruction         task (classification)         rate proxy
```
- **TACNet** uses β=0.5 — the classifier guides the encoder to keep discriminative features.
- **Baseline** uses β=0.0 — purely reconstruction-focused.
- Higher **γ** = stronger rate penalty = lower BPP = more compression.

**Key result**: At equal BPP, TACNet achieves higher classification accuracy.

**Training**
```bash
python main.py --mode run_all           # full pipeline (~hours on CPU)
python main.py --mode run_all --quick   # fast test run
```
""")

    with gr.Accordion("📚 References", open=False):
        gr.Markdown("""
- Ballé et al., "End-to-end optimized image compression," *ICLR 2017*
- Ballé et al., "Variational image compression with a scale hyperprior," *IEEE TPAMI 2020*
- He et al., "Deep residual learning for image recognition," *CVPR 2016*
- Ye et al., "AccelIR: Task-Aware Image Compression," *CVPR 2023*
- Li et al., "Image Compression for Machine and Human Vision," *ECCV 2024*
""")

    gr.Markdown(
        "---\n"
        "*TACNet — Rate–Distortion–Task Compression Framework.  "
        "Built with PyTorch + Gradio.*"
    )


# ── Launch ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=False,
        css=CSS,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    )
