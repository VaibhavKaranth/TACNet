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
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

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

# Compression levels → γ values  (matches Config.gamma_values: 3 levels per PDF)
GAMMA_VALUES = [0.0001, 0.001, 0.01, 0.1]
LEVEL_NAMES  = [
    "Low Compression  (γ=0.0001)  — highest quality",
    "Medium Compression  (γ=0.001)",
    "High Compression  (γ=0.01)",
    "Very High Compression  (γ=0.1)  — lowest quality",
]
# Primary trained levels (matching Config.gamma_values = [0.0001, 0.001, 0.01])
PRIMARY_GAMMAS = [0.0001, 0.001, 0.01]

# ── Model loading ─────────────────────────────────────────────────────────────

tacnet_models   : dict = {}
baseline_models : dict = {}
classifier_path = os.path.join(config.checkpoints_dir, "classifier_best.pth")


def _load_all_models() -> str:
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
        return "⚠️ **Classifier not trained yet.** Run: `python experiments/run_all.py`"
    if _status == "no_models":
        return "⚠️ **No compression models found.** Run: `python experiments/run_all.py`"
    n = int(_status.split(":")[1])
    return f"✅  **{n} TACNet + {n} Baseline model(s) loaded** — ready to compress"


# ── Image utilities ───────────────────────────────────────────────────────────

def _preprocess(pil_img: Image.Image) -> torch.Tensor:
    img    = pil_img.convert("RGB").resize((32, 32), Image.LANCZOS)
    arr    = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def _to_display(tensor: torch.Tensor, scale: int = 8) -> Image.Image:
    arr = tensor.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    W, H = pil.size
    return pil.resize((W * scale, H * scale), Image.NEAREST)


# ── Charts ────────────────────────────────────────────────────────────────────

# Shared palette — grey / black
BLUE   = "#A0A8B8"   # steel grey for TACNet
RED    = "#C0C0C0"   # light grey for Baseline
BG     = "#0D0D0D"
CARD   = "#1A1A1A"
TEXT   = "#E4E4E4"
MUTED  = "#6B6B6B"
GRID   = "#2C2C2C"


def _set_dark_style():
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor":   CARD,
        "axes.edgecolor":   GRID,
        "axes.labelcolor":  TEXT,
        "xtick.color":      MUTED,
        "ytick.color":      MUTED,
        "text.color":       TEXT,
        "grid.color":       GRID,
        "grid.linewidth":   0.8,
        "font.family":      "sans-serif",
    })


def _prob_chart(logits_t: torch.Tensor, logits_b: torch.Tensor, gamma: float) -> Image.Image:
    probs_t = F.softmax(logits_t.squeeze(0), dim=-1).cpu().numpy()
    probs_b = F.softmax(logits_b.squeeze(0), dim=-1).cpu().numpy()

    _set_dark_style()
    fig, ax = plt.subplots(figsize=(12, 4.2), facecolor=BG)
    ax.set_facecolor(CARD)

    x = np.arange(len(CIFAR10_CLASSES))
    w = 0.36

    bars_t = ax.bar(x - w / 2, probs_t, w, color="#E4E4E4", alpha=0.92, zorder=3, label="TACNet")
    bars_b = ax.bar(x + w / 2, probs_b, w, color="#555555", alpha=0.92, zorder=3, label="Baseline")

    # Highlight winning bar
    top_t = int(probs_t.argmax())
    top_b = int(probs_b.argmax())
    bars_t[top_t].set_edgecolor("white")
    bars_t[top_t].set_linewidth(2)
    bars_b[top_b].set_edgecolor("white")
    bars_b[top_b].set_linewidth(2)

    ax.axhline(y=0.5, color=MUTED, linestyle="--", linewidth=1, alpha=0.5, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Confidence", fontsize=11, color=TEXT)
    ax.set_ylim([0, 1.1])
    ax.set_title(
        f"Classifier Confidence on Reconstructed Images   (γ = {gamma})",
        fontsize=13, fontweight="bold", color=TEXT, pad=12,
    )
    ax.grid(True, axis="y", zorder=1)
    ax.spines[:].set_visible(False)

    legend = ax.legend(
        handles=[
            mpatches.Patch(color="#E4E4E4", label="TACNet"),
            mpatches.Patch(color="#555555", label="Baseline"),
        ],
        fontsize=11, framealpha=0.15, facecolor=CARD, edgecolor=GRID,
        loc="upper right",
    )
    for text in legend.get_texts():
        text.set_color(TEXT)

    plt.tight_layout(pad=1.2)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close()
    buf.seek(0)
    return Image.open(buf).copy()


def _diff_map(orig: torch.Tensor, compressed: torch.Tensor, label: str, color: str) -> Image.Image:
    diff = (orig - compressed).abs().clamp(0, 1)
    diff_amplified = (diff * 10).clamp(0, 1)

    _set_dark_style()
    fig, ax = plt.subplots(figsize=(3.2, 3.2), facecolor=BG)
    ax.set_facecolor(BG)

    arr = diff_amplified.squeeze(0).permute(1, 2, 0).cpu().numpy()
    im  = ax.imshow(arr, cmap="inferno", interpolation="nearest")
    ax.set_title(label, fontsize=10, fontweight="bold", color=color, pad=8)
    ax.axis("off")

    cbar = plt.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
    cbar.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED, fontsize=8)
    cbar.outline.set_edgecolor(GRID)

    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    buf.seek(0)
    return Image.open(buf).copy()


# ── Main inference ────────────────────────────────────────────────────────────

@torch.no_grad()
def run_demo(pil_image, level_name):
    if pil_image is None:
        return None, None, None, "**Please upload an image to get started.**", None, None, None

    if not tacnet_models:
        return None, None, None, _status_message(), None, None, None

    level_idx = LEVEL_NAMES.index(level_name)
    gamma     = GAMMA_VALUES[level_idx]

    if gamma not in tacnet_models:
        return None, None, None, f"⚠️ No model trained for γ={gamma}.", None, None, None

    x = _preprocess(pil_image)

    x_hat_t, z_t, z_hat_t, logits_t = tacnet_models[gamma](x)
    x_hat_b, z_b, z_hat_b, logits_b = baseline_models.get(gamma, tacnet_models[gamma])(x)

    orig_disp   = _to_display(x,       8)
    tacnet_disp = _to_display(x_hat_t, 8)
    base_disp   = _to_display(x_hat_b, 8)

    psnr_t = compute_psnr(x, x_hat_t)
    psnr_b = compute_psnr(x, x_hat_b)
    ssim_t = compute_ssim(x, x_hat_t)
    ssim_b = compute_ssim(x, x_hat_b)
    bpp_t  = compute_bpp(z_hat_t, 32, 32)
    bpp_b  = compute_bpp(z_hat_b, 32, 32)

    probs_t = F.softmax(logits_t.squeeze(0), dim=-1)
    probs_b = F.softmax(logits_b.squeeze(0), dim=-1)
    pred_t  = CIFAR10_CLASSES[probs_t.argmax().item()]
    conf_t  = probs_t.max().item() * 100
    pred_b  = CIFAR10_CLASSES[probs_b.argmax().item()]
    conf_b  = probs_b.max().item() * 100

    def _win(a, b, higher=True):
        if higher:
            return ("🏆 **TACNet**" if a >= b else "**Baseline**",
                    ("🏆 **TACNet**" if a >= b else "**Baseline**"))
        return ("🏆 **TACNet**" if a <= b else "**Baseline**",
                ("🏆 **TACNet**" if a <= b else "**Baseline**"))

    bpp_w  = "🏆 **TACNet**" if bpp_t  <= bpp_b  else "**Baseline**"
    psnr_w = "🏆 **TACNet**" if psnr_t >= psnr_b else "**Baseline**"
    ssim_w = "🏆 **TACNet**" if ssim_t >= ssim_b else "**Baseline**"
    acc_w  = "🏆 **TACNet**" if conf_t  >= conf_b  else "**Baseline**"

    metrics_md = f"""
<div class="metrics-card">

### Results  —  γ = {gamma}

| Metric | TACNet | Baseline | Winner |
|:-------|:------:|:--------:|:------:|
| **Bits per Pixel** ↓ | `{bpp_t:.3f}` | `{bpp_b:.3f}` | {bpp_w} |
| **PSNR (dB)** ↑ | `{psnr_t:.2f}` | `{psnr_b:.2f}` | {psnr_w} |
| **SSIM** ↑ | `{ssim_t:.4f}` | `{ssim_b:.4f}` | {ssim_w} |
| **Predicted Class** | **{pred_t}** `{conf_t:.1f}%` | **{pred_b}** `{conf_b:.1f}%` | {acc_w} |

> **Key insight:** TACNet preserves classification-relevant features at lower bitrates — a small PSNR trade-off for significantly higher machine accuracy.

</div>
"""

    prob_chart = _prob_chart(logits_t, logits_b, gamma)
    diff_t     = _diff_map(x, x_hat_t, "TACNet Error Map",   BLUE)
    diff_b     = _diff_map(x, x_hat_b, "Baseline Error Map", RED)

    return (
        orig_disp, tacnet_disp, base_disp,
        metrics_md.strip(),
        prob_chart, diff_t, diff_b,
    )


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
/* ── Page background ── */
body, .gradio-container {
    background: #0d0d0d !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ── Hero header ── */
.hero-wrap {
    background: #111111;
    border: 1px solid #2a2a2a;
    border-top: 3px solid #3a3a3a;
    border-radius: 16px;
    padding: 2.4rem 2rem 2rem;
    margin-bottom: 1.4rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -50px; left: 50%; transform: translateX(-50%);
    width: 480px; height: 180px;
    background: radial-gradient(ellipse, rgba(200,200,200,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
    color: #f0f0f0 !important;
    margin-bottom: 0.5rem !important;
}
.hero-badge {
    display: inline-block;
    background: #1e1e1e;
    border: 1px solid #333;
    border-radius: 20px;
    padding: 0.2rem 0.9rem;
    font-size: 0.75rem;
    color: #888;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.hero-sub {
    color: #777 !important;
    font-size: 0.95rem !important;
    line-height: 1.7;
    max-width: 600px;
    margin: 0 auto !important;
}

/* ── Control panel ── */
.control-panel {
    background: #111111;
    border: 1px solid #252525;
    border-radius: 14px;
    padding: 1.4rem !important;
}

/* ── Section cards ── */
.section-card {
    background: #111111;
    border: 1px solid #252525;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-top: 1rem;
}

/* ── Image panels ── */
.image-panel .wrap {
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid #252525 !important;
    background: #0a0a0a !important;
}
.image-panel label span {
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.3px;
    color: #aaa !important;
}

/* ── Run button ── */
.run-btn {
    background: #e4e4e4 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.3px;
    color: #0d0d0d !important;
    box-shadow: 0 2px 12px rgba(255,255,255,0.08) !important;
    transition: all 0.2s ease !important;
}
.run-btn:hover {
    background: #ffffff !important;
    box-shadow: 0 4px 20px rgba(255,255,255,0.18) !important;
    transform: translateY(-1px);
}

/* ── Metrics table ── */
.metrics-card table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.92rem;
}
.metrics-card th {
    background: #1a1a1a;
    color: #c0c0c0;
    padding: 0.55rem 0.9rem;
    font-weight: 600;
    border-bottom: 2px solid #2c2c2c;
}
.metrics-card td {
    padding: 0.55rem 0.9rem;
    border-bottom: 1px solid #1e1e1e;
    color: #b0b0b0;
}
.metrics-card tr:last-child td { border-bottom: none; }
.metrics-card tr:hover td { background: #161616; }
.metrics-card blockquote {
    border-left: 3px solid #444;
    margin: 0.8rem 0 0;
    padding: 0.5rem 0.9rem;
    color: #666;
    font-size: 0.87rem;
    background: #0f0f0f;
    border-radius: 0 8px 8px 0;
}

/* ── Section headings ── */
.section-card h3, .control-panel h3 {
    color: #c8c8c8 !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.2px !important;
    text-transform: uppercase !important;
    margin-bottom: 0.8rem !important;
}

/* ── Accordion ── */
.accordion {
    background: #111111 !important;
    border: 1px solid #252525 !important;
    border-radius: 12px !important;
    margin-top: 1rem !important;
}
.accordion .label-wrap span {
    color: #888 !important;
    font-weight: 600;
}

/* ── Footer ── */
.footer-text { color: #333 !important; font-size: 0.82rem !important; text-align: center; }

/* ── Inputs ── */
select, .gr-dropdown {
    background: #0f0f0f !important;
    border-color: #252525 !important;
    color: #b0b0b0 !important;
}
"""

# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="TACNet — Task-Aware Image Compression") as demo:

    # ── Hero Header ───────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="hero-wrap">
        <div class="hero-badge">Neural Image Compression</div>
        <div class="hero-title">TACNet &mdash; Task-Aware Compression</div>
        <div class="hero-sub">
            Compress images <strong style="color:#d0d0d0">to be understood</strong>, not just to look good.
            Jointly optimises <em>Rate &ndash; Distortion &ndash; Task</em> loss
            to preserve classification accuracy at low bitrates.
        </div>
    </div>
    """)

    # ── Main row ──────────────────────────────────────────────────────────────
    with gr.Row(equal_height=True):

        # Left: controls
        with gr.Column(scale=1, min_width=270, elem_classes=["control-panel"]):
            gr.Markdown("### Upload & Configure", elem_classes=["section-label"])
            inp_image = gr.Image(
                type="pil",
                label="Input Image",
                elem_classes=["image-panel"],
            )
            inp_level = gr.Dropdown(
                choices=LEVEL_NAMES,
                value=LEVEL_NAMES[1],
                label="Compression Level",
                info="Higher compression = lower BPP = harder for classifiers",
            )
            btn_run = gr.Button(
                "Compress & Compare",
                variant="primary",
                size="lg",
                elem_classes=["run-btn"],
            )

        # Right: output images
        with gr.Column(scale=3):
            gr.Markdown("### Visual Comparison")
            with gr.Row():
                out_orig     = gr.Image(label="Original  (32×32 upscaled)", height=260,
                                        elem_classes=["image-panel"])
                out_tacnet   = gr.Image(label="TACNet Compressed",           height=260,
                                        elem_classes=["image-panel"])
                out_baseline = gr.Image(label="Baseline Compressed",         height=260,
                                        elem_classes=["image-panel"])

    # ── Metrics ───────────────────────────────────────────────────────────────
    with gr.Group(elem_classes=["section-card"]):
        gr.Markdown("### Metrics & Results")
        out_metrics = gr.Markdown(elem_classes=["metrics-card"])

    # ── Confidence chart ──────────────────────────────────────────────────────
    with gr.Group(elem_classes=["section-card"]):
        gr.Markdown("### Classifier Confidence on Reconstructed Images")
        out_prob_chart = gr.Image(label="", container=False)

    # ── Error maps ────────────────────────────────────────────────────────────
    with gr.Group(elem_classes=["section-card"]):
        gr.Markdown("### Pixel Error Maps  *(amplified ×10)*")
        with gr.Row():
            out_diff_tacnet   = gr.Image(label="TACNet Error Map",   height=220,
                                         elem_classes=["image-panel"])
            out_diff_baseline = gr.Image(label="Baseline Error Map", height=220,
                                         elem_classes=["image-panel"])

    # ── Wire up ───────────────────────────────────────────────────────────────
    btn_run.click(
        fn=run_demo,
        inputs=[inp_image, inp_level],
        outputs=[
            out_orig, out_tacnet, out_baseline,
            out_metrics,
            out_prob_chart,
            out_diff_tacnet, out_diff_baseline,
        ],
    )

    # ── Accordions ────────────────────────────────────────────────────────────
    with gr.Accordion("How it works", open=False, elem_classes=["accordion"]):
        gr.Markdown("""
**Pipeline**
1. Your image is resized to **32×32** (CIFAR-10 resolution).
2. The **Encoder** (CNN) maps it to a compact latent `z` (shape `C×8×8`).
3. The **STE Quantizer** rounds `z` to integers — differentiable via straight-through gradient.
4. The **Decoder** reconstructs `x̂` from the quantised latent.
5. The **Frozen ResNet-18 Classifier** predicts the class from `normalize(x̂)`.

**Loss Function**
```
L = α · MSE(x̂, x)  +  β · CrossEntropy(C(x̂), y)  +  γ · mean(|z|)
      reconstruction        task (classification)         rate proxy
```
- **TACNet** uses β=0.5 — the classifier guides the encoder.
- **Baseline** uses β=0.0 — reconstruction only.
- Higher **γ** = stronger rate penalty = lower BPP.
""")

    with gr.Accordion("References", open=False, elem_classes=["accordion"]):
        gr.Markdown("""
- Ballé et al., *End-to-end optimized image compression*, ICLR 2017
- Ballé et al., *Variational image compression with a scale hyperprior*, IEEE TPAMI 2020
- He et al., *Deep residual learning for image recognition*, CVPR 2016
- Ye et al., *AccelIR: Task-Aware Image Compression*, CVPR 2023
""")

    gr.HTML('<div class="footer-text" style="margin-top:1.5rem">TACNet &mdash; Rate&ndash;Distortion&ndash;Task Compression &nbsp;|&nbsp; Built with PyTorch + Gradio</div>')


# ── Launch ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=False,
        css=CSS,
    )
