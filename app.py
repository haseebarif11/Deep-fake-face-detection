import time
import numpy as np
from PIL import Image, ImageFilter
import gradio as gr
import torch

# Modular imports from src/
from src.inference import DeepFakePredictor

# Global model (loaded once at startup)
_predictor = DeepFakePredictor(model_path="best_model_rebuilt.pth", threshold=0.5)


def analyze_image_properties(image):
    img_array = np.array(image)
    gray = np.mean(img_array, axis=2)
    noise_level = float(np.std(np.diff(gray.flatten()[:10000])))
    noise_score = min(100, noise_level * 4)
    r_std = float(np.std(img_array[:, :, 0]))
    g_std = float(np.std(img_array[:, :, 1]))
    b_std = float(np.std(img_array[:, :, 2]))
    color_score = min(100, (r_std + g_std + b_std) / 3.0 / 2.8)
    blurred = np.array(image.filter(ImageFilter.FIND_EDGES))
    sharpness = float(np.var(blurred))
    sharp_score = min(100, sharpness / 25.0)
    hist = np.histogram(gray, bins=64)[0]
    hist_norm = hist / hist.sum()
    hist_norm = hist_norm[hist_norm > 0]
    entropy = float(-np.sum(hist_norm * np.log2(hist_norm)))
    entropy_score = max(0, min(100, 100 - (entropy * 10.0)))
    w = image.width
    left = np.array(image.crop((0, 0, w // 2, image.height)).resize((64, 64)))
    right = np.array(image.crop((w // 2, 0, w, image.height)).resize((64, 64)))
    right_flipped = right[:, ::-1, :]
    symmetry_diff = float(np.mean(np.abs(left.astype(float) - right_flipped.astype(float))))
    symmetry_score = max(0, min(100, 100 - symmetry_diff * 1.2))
    return {
        "noise":     round(noise_score, 1),
        "color":     round(color_score, 1),
        "sharpness": round(sharp_score, 1),
        "entropy":   round(entropy_score, 1),
        "symmetry":  round(symmetry_score, 1),
    }


def get_image_info(image):
    w, h = image.size
    arr = np.array(image)
    return {
        "width": w, "height": h, "mode": image.mode,
        "brightness": round(float(np.mean(arr)), 1),
        "aspect": f"{round(w / h, 2)}:1",
        "megapixels": round((w * h) / 1_000_000, 2),
    }


def predict_image(image):
    if image is None:
        return _await_html()

    t0 = time.time()
    image = image.convert("RGB")

    result = _predictor.predict(image, detect_face=True)

    fake_prob = result["fake_probability"] / 100.0
    real_prob = result["real_probability"] / 100.0
    confidence = result["confidence_percentage"]
    prediction_label = "real" if result["predicted_class"] == "Authentic" else "fake"
    face_detected = result["face_detected"]
    inference_ms = round((time.time() - t0) * 1000, 1)
    device_label = "GPU" if _predictor.device.type == "cuda" else "CPU"

    if confidence < 75:
        certainty = "UNCERTAIN"; certainty_color = "#94A3B8"; is_uncertain = True
    elif confidence < 85:
        certainty = "MODERATE"; certainty_color = "#FBBF24"; is_uncertain = False
    else:
        certainty = "HIGH"
        certainty_color = "#F87171" if prediction_label == "fake" else "#34D399"
        is_uncertain = False

    signals = analyze_image_properties(image)
    img_info = get_image_info(image)

    if is_uncertain:
        verdict_color = "#94A3B8"; verdict_bg = "rgba(148,163,184,0.08)"
        verdict_border = "rgba(148,163,184,0.35)"; ring_shadow = "rgba(148,163,184,0.3)"
        verdict_text = "UNCERTAIN"; verdict_icon = "?"
    elif prediction_label == "fake":
        verdict_color = "#EF4444"; verdict_bg = "rgba(239,68,68,0.08)"
        verdict_border = "rgba(239,68,68,0.35)"; ring_shadow = "rgba(239,68,68,0.3)"
        verdict_text = "AI-GENERATED"; verdict_icon = "warning"
    else:
        verdict_color = "#10B981"; verdict_bg = "rgba(16,185,129,0.08)"
        verdict_border = "rgba(16,185,129,0.35)"; ring_shadow = "rgba(16,185,129,0.3)"
        verdict_text = "AUTHENTIC"; verdict_icon = "check"

    if is_uncertain:
        interp = f"Result is inconclusive, model confidence ({confidence:.1f}%) is below 75% threshold. Manual review recommended."
    elif prediction_label == "fake" and confidence >= 85:
        interp = "Strong indicators of AI synthesis detected across multiple signal channels."
    elif prediction_label == "fake":
        interp = "Several patterns consistent with generative model artifacts were detected."
    elif confidence >= 85:
        interp = "No significant synthetic artifacts detected. Image appears camera-captured."
    else:
        interp = "Mostly authentic characteristics with minor ambiguous regions."

    face_badge = (
        '<span style="font-family:JetBrains Mono,monospace;font-size:9px;letter-spacing:1.5px;color:#10B981;padding:2px 8px;border:1px solid rgba(16,185,129,.35);border-radius:4px;background:rgba(16,185,129,.08)">FACE DETECTED</span>'
        if face_detected else
        '<span style="font-family:JetBrains Mono,monospace;font-size:9px;letter-spacing:1.5px;color:#FBBF24;padding:2px 8px;border:1px solid rgba(251,191,36,.35);border-radius:4px;background:rgba(251,191,36,.08)">NO FACE - FULL IMAGE</span>'
    )

    fake_fill_w = f"{fake_prob * 100:.1f}%"
    real_fill_w = f"{real_prob * 100:.1f}%"

    def sig_bar(label, val, tooltip):
        color = "#EF4444" if val > 70 else "#FBBF24" if val > 40 else "#10B981"
        return (
            f'<div class="sig-row" title="{tooltip}">'
            f'<span class="sig-name">{label}</span>'
            f'<div class="sig-track"><div class="sig-fill" style="width:{val}%;background:{color};box-shadow:0 0 6px {color}55"></div></div>'
            f'<span class="sig-val" style="color:{color}">{val}</span>'
            f'</div>'
        )

    sig_html = (
        sig_bar("NOISE PATTERN", signals["noise"], "Irregular noise may indicate GAN artifacts") +
        sig_bar("COLOR DIST.", signals["color"], "Color distribution variance across channels") +
        sig_bar("SHARPNESS", signals["sharpness"], "Unnatural sharpness can indicate synthesis") +
        sig_bar("PIXEL ENTROPY", signals["entropy"], "Low entropy (smooth pixels) suggests AI generation") +
        sig_bar("FACE SYMMETRY", signals["symmetry"], "AI faces tend to be unusually symmetric")
    )

    def gauge_svg(pct, color):
        r = 36; cx = 44; cy = 44
        circ = 2 * 3.14159 * r
        dash = circ * pct / 100
        return (
            f'<svg width="88" height="88" viewBox="0 0 88 88">'
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="7"/>'
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="7" '
            f'stroke-dasharray="{dash:.1f} {circ:.1f}" stroke-dashoffset="{circ/4:.1f}" '
            f'stroke-linecap="round" style="filter:drop-shadow(0 0 4px {color})"/>'
            f'<text x="{cx}" y="{cy+5}" text-anchor="middle" font-family="JetBrains Mono,monospace" '
            f'font-size="13" font-weight="700" fill="{color}">{pct:.0f}%</text>'
            f'</svg>'
        )

    fake_gauge = gauge_svg(fake_prob * 100, "#EF4444")
    real_gauge = gauge_svg(real_prob * 100, "#10B981")

    html = (
        '<div class="result-wrap">'
        f'<div class="verdict-banner" style="background:{verdict_bg};border:1px solid {verdict_border}">'
        f'<div class="verdict-ring" style="border-color:{verdict_color};box-shadow:0 0 22px {ring_shadow}">'
        f'<span style="font-size:24px">{verdict_icon}</span></div>'
        '<div class="verdict-body">'
        '<div class="v-eyebrow">FORENSIC VERDICT</div>'
        f'<div class="v-main" style="color:{verdict_color}">{verdict_text}</div>'
        f'<div class="v-sub">{interp}</div>'
        f'<div style="margin-top:8px">{face_badge}</div>'
        '</div>'
        f'<div class="verdict-badge" style="background:{verdict_color}22;border:1px solid {verdict_color}55;color:{verdict_color}">'
        f'<span class="badge-conf">{confidence:.1f}%</span>'
        f'<span class="badge-tier" style="color:{certainty_color}">{certainty}</span>'
        '</div></div>'
        '<div class="metrics-row">'
        f'<div class="gauge-card"><div class="gauge-label">SYNTHETIC PROB.</div>{fake_gauge}</div>'
        f'<div class="gauge-card"><div class="gauge-label">AUTHENTIC PROB.</div>{real_gauge}</div>'
        '<div class="bars-card">'
        '<div class="prob-row">'
        '<span class="prob-name fake-col">SYNTHETIC</span>'
        '<div class="prob-track">'
        f'<div class="prob-fill" style="width:{fake_fill_w};background:linear-gradient(90deg,#7F1D1D,#EF4444);box-shadow:0 0 6px rgba(239,68,68,.4)"></div>'
        '</div>'
        f'<span class="prob-val fake-col">{fake_prob*100:.1f}%</span>'
        '</div>'
        '<div class="prob-row">'
        '<span class="prob-name real-col">AUTHENTIC</span>'
        '<div class="prob-track">'
        f'<div class="prob-fill" style="width:{real_fill_w};background:linear-gradient(90deg,#064E3B,#10B981);box-shadow:0 0 6px rgba(16,185,129,.4)"></div>'
        '</div>'
        f'<span class="prob-val real-col">{real_prob*100:.1f}%</span>'
        '</div>'
        '<div class="infer-row">'
        '<span class="infer-label">INFERENCE TIME</span>'
        f'<span class="infer-val">{inference_ms} ms</span>'
        '</div></div></div>'
        '<div class="section-card">'
        '<div class="card-header"><span class="card-title">FORENSIC SIGNAL ANALYSIS</span>'
        '<span class="card-hint">Higher = more suspicious</span></div>'
        f'<div class="sig-grid">{sig_html}</div>'
        '</div>'
        '<div class="section-card">'
        '<div class="card-header"><span class="card-title">IMAGE METADATA</span></div>'
        '<div class="meta-grid">'
        f'<div class="meta-item"><span class="meta-k">DIMENSIONS</span><span class="meta-v">{img_info["width"]} x {img_info["height"]} px</span></div>'
        f'<div class="meta-item"><span class="meta-k">MEGAPIXELS</span><span class="meta-v">{img_info["megapixels"]} MP</span></div>'
        f'<div class="meta-item"><span class="meta-k">ASPECT RATIO</span><span class="meta-v">{img_info["aspect"]}</span></div>'
        f'<div class="meta-item"><span class="meta-k">COLOR MODE</span><span class="meta-v">{img_info["mode"]}</span></div>'
        f'<div class="meta-item"><span class="meta-k">BRIGHTNESS</span><span class="meta-v">{img_info["brightness"]}/255</span></div>'
        f'<div class="meta-item"><span class="meta-k">DEVICE</span><span class="meta-v">{device_label}</span></div>'
        '</div></div>'
        '<div class="scan-footer">'
        '<span>MODEL - EfficientNet-B3</span>'
        '<span>FACE DET. - MTCNN</span>'
        '<span>FRAMEWORK - PyTorch</span>'
        '<span>VERSION - 2.0.0</span>'
        '</div></div>'
    )
    return html


def _await_html():
    return (
        '<div class="await-state">'
        '<div class="await-icon">hex</div>'
        '<div class="await-text">AWAITING SCAN</div>'
        '<div class="await-sub">Upload an image and run forensic scan</div>'
        '</div>'
    )


custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body,.gradio-container{background:#080A14 !important;font-family:'Space Grotesk',sans-serif !important;color:#E2E8F0 !important;min-height:100vh;}
.gradio-container{max-width:1200px !important;width:100% !important;margin:0 auto !important;padding:0 24px 60px !important;}
.contain{max-width:1200px !important}
.app-header{text-align:center;padding:44px 0 32px;position:relative}
.app-eyebrow{font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:4px;color:#06B6D4;margin-bottom:12px}
.app-title{font-size:clamp(32px,5vw,54px);font-weight:700;letter-spacing:-2px;line-height:1;background:linear-gradient(135deg,#E2E8F0 0%,#A78BFA 55%,#06B6D4 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:10px}
.app-title-sub{display:block;font-size:clamp(14px,2vw,18px);font-weight:300;letter-spacing:6px;color:#475569;margin-top:6px}
.app-subtitle{font-size:14px;color:#475569;max-width:560px;margin:12px auto 0;line-height:1.7}
.stats-bar{display:flex;justify-content:center;gap:32px;margin:24px 0 0;flex-wrap:wrap}
.stat-item{display:flex;flex-direction:column;align-items:center;gap:2px}
.stat-num{font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;color:#A78BFA}
.stat-lbl{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:2px;color:#334155}
.glow-orb{position:fixed;border-radius:50%;filter:blur(100px);pointer-events:none;z-index:-1;opacity:.09}
.orb-v{width:500px;height:500px;background:#7C3AED;top:-120px;left:-120px}
.orb-c{width:360px;height:360px;background:#06B6D4;bottom:-60px;right:-60px}
.orb-m{width:280px;height:280px;background:#EC4899;top:40%;right:20%}
.main-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start}
@media(max-width:780px){.main-grid{grid-template-columns:1fr}}
.sec-label{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:3px;color:#334155;margin-bottom:6px}
.gradio-container [data-testid="image"]{border:1.5px dashed rgba(124,58,237,.4) !important;border-radius:16px !important;background:rgba(10,12,24,.7) !important;transition:all .25s !important;min-height:340px !important;}
.gradio-container [data-testid="image"]:hover{border-color:rgba(124,58,237,.75) !important;background:rgba(124,58,237,.05) !important;}
#scan-btn{width:100% !important;padding:15px !important;border-radius:12px !important;border:none !important;background:linear-gradient(135deg,#7C3AED 0%,#4F46E5 50%,#0891B2 100%) !important;color:#fff !important;font-family:'Space Grotesk',sans-serif !important;font-size:14px !important;font-weight:600 !important;letter-spacing:2px !important;text-transform:uppercase !important;cursor:pointer !important;box-shadow:0 0 28px rgba(124,58,237,.35) !important;transition:all .2s !important;}
#scan-btn:hover{transform:translateY(-2px) !important;box-shadow:0 0 44px rgba(124,58,237,.55) !important}
#scan-btn:active{transform:translateY(0) !important}
.gradio-container button.secondary{background:rgba(15,20,40,.8) !important;border:1px solid rgba(124,58,237,.2) !important;color:#475569 !important;border-radius:10px !important;font-family:'Space Grotesk',sans-serif !important;transition:all .2s !important;}
.gradio-container button.secondary:hover{border-color:rgba(124,58,237,.5) !important;color:#E2E8F0 !important}
.info-cards{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.info-card{background:rgba(15,18,32,.7);border:1px solid rgba(255,255,255,.06);border-radius:12px;padding:12px 14px}
.info-card-title{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:2px;color:#334155;margin-bottom:6px}
.info-card-val{font-family:'JetBrains Mono',monospace;font-size:13px;color:#7C3AED;font-weight:600}
.info-card-desc{font-size:11px;color:#475569;margin-top:3px;line-height:1.5}
.gradio-container .prose,.gradio-html{background:transparent !important;border:none !important;padding:0 !important}
.await-state{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:80px 20px;gap:12px;border:1px dashed rgba(124,58,237,.2);border-radius:16px;background:rgba(10,12,24,.5)}
.await-icon{font-size:36px;opacity:.2}
.await-text{font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:4px;color:#334155}
.await-sub{font-size:12px;color:#1E293B;letter-spacing:.5px}
.result-wrap{display:flex;flex-direction:column;gap:12px}
.verdict-banner{border-radius:14px;padding:18px 20px;display:flex;align-items:center;gap:16px;position:relative;overflow:hidden}
.verdict-ring{width:56px;height:56px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0;border:2px solid;background:rgba(0,0,0,.2)}
.verdict-body{flex:1}
.v-eyebrow{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:3px;color:#475569;margin-bottom:3px}
.v-main{font-size:22px;font-weight:700;letter-spacing:-.5px;margin-bottom:4px}
.v-sub{font-size:12px;color:#64748B;line-height:1.5;max-width:280px}
.verdict-badge{border-radius:10px;padding:8px 14px;display:flex;flex-direction:column;align-items:center;flex-shrink:0}
.badge-conf{font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700;line-height:1}
.badge-tier{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:2px;margin-top:2px}
.metrics-row{display:flex;gap:10px}
.gauge-card{background:rgba(10,11,20,.7);border:1px solid rgba(255,255,255,.06);border-radius:12px;padding:12px 14px;display:flex;flex-direction:column;align-items:center;gap:4px;flex:0 0 auto}
.gauge-label{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:2px;color:#334155;text-align:center}
.bars-card{flex:1;background:rgba(10,11,20,.7);border:1px solid rgba(255,255,255,.06);border-radius:12px;padding:14px 16px;display:flex;flex-direction:column;gap:10px;justify-content:center}
.prob-row{display:flex;align-items:center;gap:10px}
.prob-name{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:2px;width:72px;flex-shrink:0}
.prob-track{flex:1;height:5px;background:rgba(255,255,255,.06);border-radius:999px;overflow:hidden}
.prob-fill{height:100%;border-radius:999px;transition:width .8s cubic-bezier(.4,0,.2,1)}
.prob-val{font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:600;width:42px;text-align:right;flex-shrink:0}
.fake-col{color:#F87171}.real-col{color:#34D399}
.infer-row{display:flex;justify-content:space-between;padding-top:8px;border-top:1px solid rgba(255,255,255,.05);margin-top:2px}
.infer-label{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:2px;color:#334155}
.infer-val{font-family:'JetBrains Mono',monospace;font-size:10px;color:#7C3AED}
.section-card{background:rgba(10,11,20,.7);border:1px solid rgba(255,255,255,.06);border-radius:12px;padding:14px 16px}
.card-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
.card-title{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:3px;color:#475569}
.card-hint{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:1px;color:#1E293B}
.sig-grid{display:flex;flex-direction:column;gap:9px}
.sig-row{display:flex;align-items:center;gap:10px;cursor:help}
.sig-name{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:1.5px;color:#475569;width:90px;flex-shrink:0}
.sig-track{flex:1;height:4px;background:rgba(255,255,255,.05);border-radius:999px;overflow:hidden}
.sig-fill{height:100%;border-radius:999px;transition:width .6s ease}
.sig-val{font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;width:30px;text-align:right;flex-shrink:0}
.meta-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}
.meta-item{display:flex;flex-direction:column;gap:2px}
.meta-k{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:1.5px;color:#334155}
.meta-v{font-family:'JetBrains Mono',monospace;font-size:11px;color:#94A3B8;font-weight:500}
.scan-footer{display:flex;gap:16px;flex-wrap:wrap;padding:10px 0 0}
.scan-footer span{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:1.5px;color:#1E293B}
.disclaimer{margin-top:20px;padding:14px 18px;background:rgba(15,18,32,.5);border:1px solid rgba(255,255,255,.04);border-radius:10px;font-family:'JetBrains Mono',monospace;font-size:10px;color:#1E293B;line-height:1.8}
.disclaimer strong{color:#334155}
.gradio-container label{font-family:'JetBrains Mono',sans-serif !important;color:#334155 !important;font-size:10px !important;letter-spacing:2px !important;text-transform:uppercase !important;}
footer{display:none !important}
"""

header_html = """
<div class="glow-orb orb-v"></div>
<div class="glow-orb orb-c"></div>
<div class="glow-orb orb-m"></div>
<div class="app-header">
  <div class="app-eyebrow">// forensic image analysis - v2.0</div>
  <div class="app-title">DeepFake Detector
    <span class="app-title-sub">FORENSIC - NEURAL - ANALYSIS</span>
  </div>
  <div class="app-subtitle">
    EfficientNet-B3 + MTCNN face detection -- classifies AI-generated vs. authentic images.
    Includes real-time forensic signal extraction and image metadata analysis.
  </div>
  <div class="stats-bar">
    <div class="stat-item"><span class="stat-num">B3</span><span class="stat-lbl">EfficientNet</span></div>
    <div class="stat-item"><span class="stat-num">300px</span><span class="stat-lbl">Input Res.</span></div>
    <div class="stat-item"><span class="stat-num">MTCNN</span><span class="stat-lbl">Face Detect.</span></div>
    <div class="stat-item"><span class="stat-num">5</span><span class="stat-lbl">Signal Channels</span></div>
    <div class="stat-item"><span class="stat-num">2</span><span class="stat-lbl">Classes</span></div>
  </div>
</div>
"""

info_cards_html = """
<div class="info-cards">
  <div class="info-card">
    <div class="info-card-title">MODEL ARCHITECTURE</div>
    <div class="info-card-val">EfficientNet-B3</div>
    <div class="info-card-desc">Compound-scaled CNN with dual dropout classifier head</div>
  </div>
  <div class="info-card">
    <div class="info-card-title">FACE DETECTION</div>
    <div class="info-card-val">MTCNN</div>
    <div class="info-card-desc">Multi-task Cascaded CNN - auto face crop and align</div>
  </div>
  <div class="info-card">
    <div class="info-card-title">FORENSIC SIGNALS</div>
    <div class="info-card-val">5 Channels</div>
    <div class="info-card-desc">Noise, color, sharpness, entropy, face symmetry</div>
  </div>
  <div class="info-card">
    <div class="info-card-title">CONFIDENCE THRESHOLD</div>
    <div class="info-card-val">less than 75% = Uncertain</div>
    <div class="info-card-desc">Results below 75% confidence are flagged as inconclusive</div>
  </div>
</div>
"""

disclaimer_html = """
<div class="disclaimer">
  <strong>DISCLAIMER</strong> This tool is a detection aid based on statistical patterns and is not a forensic-grade instrument.
  Results may vary across image types, compression levels, and generation methods. Forensic signals are heuristic-based approximations.
  Do not use as sole evidence in any formal process.
</div>
"""

with gr.Blocks(title="DeepFake Detector", css=custom_css) as demo:
    gr.HTML(header_html)

    with gr.Row(elem_classes=["main-grid"]):
        with gr.Column(scale=1):
            gr.HTML('<div class="sec-label">Input - Upload Image</div>')
            image_input = gr.Image(type="pil", label="", show_label=False, height=340)
            scan_btn = gr.Button("RUN FORENSIC SCAN", variant="primary", elem_id="scan-btn")
            gr.HTML(info_cards_html)

        with gr.Column(scale=1):
            gr.HTML('<div class="sec-label">Output - Analysis Result</div>')
            result_output = gr.HTML(
                value='<div class="await-state"><div class="await-icon">hex</div><div class="await-text">AWAITING SCAN</div><div class="await-sub">Upload an image and run forensic scan</div></div>',
                show_label=False,
            )

    gr.HTML(disclaimer_html)
    scan_btn.click(fn=predict_image, inputs=[image_input], outputs=[result_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
