"""
Blind detection: heuristic pipeline + optional trained EfficientNet-B4.

When checkpoints/efficientnet_b4_fake.pth exists (3-class model), the trained
model drives the prediction. Heuristic signals remain active and are reported
in the analysis breakdown for explainability.

Prediction classes: REAL | AI_GENERATED | EDITED

Heuristic signals (used when model absent, or for analysis breakdown):
  1. Spectral decay   — natural images follow 1/f^alpha law; AI deviates
  2. GAN periodic     — upsampling checkerboard at N/2 frequencies
  3. Noise residual   — AI images are unnaturally smooth
  4. Texture          — AI portraits have suspiciously uniform local texture
  5. Edge/gradient    — AI images have abnormally sharp, uniform edges

Heuristic class logic (no model):
  composite >= 0.45 → FAKE
    GAN-heavy (gan > spectral) → AI_GENERATED
    Edit-heavy (noise/texture) → EDITED
  composite < 0.45 → REAL
"""

import io
import sys
import uuid
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

# ─── load trained CNN model (optional) ─────────────────────────────────────────

_MODEL = None
_TRANSFORM = None

def _try_load_cnn():
    global _MODEL, _TRANSFORM
    checkpoint = Path(__file__).parent.parent / "checkpoints" / "efficientnet_b4_fake.pth"
    if not checkpoint.exists():
        return

    try:
        import torch
        import torchvision.transforms as T
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from models.cnn_model import FakeImageDetector

        state = torch.load(checkpoint, map_location="cpu")
        model = FakeImageDetector(pretrained=False)
        model.load_state_dict(state["model"] if "model" in state else state)
        model.eval()

        _MODEL = model
        _TRANSFORM = T.Compose([
            T.Resize((380, 380)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        val_f1 = state.get("val_macro_f1", state.get("val_f1", "?"))
        print(f"[FakeGuard] Loaded 3-class CNN (val_macro_f1={val_f1})")
    except Exception as e:
        print(f"[FakeGuard] CNN load failed ({e}), using heuristics only")


_try_load_cnn()

STATIC_DIR = Path(__file__).parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

CLASSES = ["REAL", "AI_GENERATED", "EDITED"]


# ─── helpers ───────────────────────────────────────────────────────────────────

def _load_image(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img)


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)


# ─── signal 1: spectral decay ───────────────────────────────────────────────────

def _spectral_score(img: np.ndarray) -> float:
    """
    Natural images: log radial spectrum decays ~linearly (1/f^alpha, alpha~2).
    AI images: flatter high-frequency shelf, deviating from this linear decay.
    Returns deviation from expected decay → higher = more suspicious.
    """
    gray = _to_gray(img)
    h, w = gray.shape
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.ogrid[:h, :w]
    radius = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2).astype(int)
    max_r = min(cx, cy) - 1

    radial_sum = np.zeros(max_r + 1)
    radial_cnt = np.zeros(max_r + 1)
    mask = radius <= max_r
    np.add.at(radial_sum, radius[mask], magnitude[mask])
    np.add.at(radial_cnt, radius[mask], 1)
    radial_cnt[radial_cnt == 0] = 1
    radial_profile = np.log1p(radial_sum[1:] / radial_cnt[1:])

    freqs = np.linspace(0, 1, len(radial_profile))
    if radial_profile.max() > 0:
        profile_norm = radial_profile / radial_profile.max()
    else:
        return 0.0

    coeffs = np.polyfit(freqs, profile_norm, 1)
    fitted = np.polyval(coeffs, freqs)
    residuals = profile_norm - fitted

    hf_half = len(residuals) // 2
    hf_excess = float(np.clip(residuals[hf_half:].mean() + 0.5, 0, 1))
    irregularity = float(np.clip(residuals.std() * 4, 0, 1))

    return float(np.clip(0.6 * hf_excess + 0.4 * irregularity, 0, 1))


# ─── signal 2: GAN periodic peaks ──────────────────────────────────────────────

def _gan_periodic_score(img: np.ndarray) -> float:
    """
    GAN transposed-convolution upsampling creates checkerboard artifacts visible
    as spectral peaks at multiples of N/2 in the FFT magnitude.
    """
    gray = _to_gray(img)
    h, w = gray.shape

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))

    cy, cx = h // 2, w // 2
    nh, nw = h // 4, w // 4

    def patch_mean(r, c, hw=3):
        return mag[max(0, r-hw):r+hw, max(0, c-hw):c+hw].mean()

    artifact_energy = np.mean([
        patch_mean(cy - nh, cx - nw),
        patch_mean(cy - nh, cx + nw),
        patch_mean(cy + nh, cx - nw),
        patch_mean(cy + nh, cx + nw),
    ])
    center_energy = patch_mean(cy, cx, hw=5) + 1e-8
    peak_ratio = float(np.clip(artifact_energy / center_energy, 0, 1))

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx**2 + gy**2)
    flat = gmag.flatten()
    if len(flat) > 4:
        lag1 = float(np.corrcoef(flat[:-1], flat[1:])[0, 1])
        lag2 = float(np.corrcoef(flat[:-2], flat[2:])[0, 1])
        periodicity = float(np.clip(abs(lag2) - abs(lag1) + 0.5, 0, 1))
    else:
        periodicity = 0.0

    return float(np.clip(0.5 * peak_ratio + 0.5 * periodicity, 0, 1))


# ─── signal 3: noise residual ──────────────────────────────────────────────────

def _noise_residual_score(img: np.ndarray) -> float:
    """
    Real cameras produce spatially correlated photon + sensor noise (PRNU).
    AI-generated images are unnaturally smooth — their noise residual energy
    is anomalously low or has flat spatial frequency distribution.
    Edited images can show noise inconsistency between regions.
    """
    gray = _to_gray(img)

    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    noise_sigma = np.median(np.abs(laplacian)) / 0.6745

    low_noise_score = float(np.clip(1.0 - noise_sigma / 10.0, 0, 1))

    blocks = []
    bh, bw = max(gray.shape[0] // 8, 1), max(gray.shape[1] // 8, 1)
    for i in range(0, gray.shape[0] - bh, bh):
        for j in range(0, gray.shape[1] - bw, bw):
            block = laplacian[i:i+bh, j:j+bw]
            blocks.append(np.abs(block).mean())
    if len(blocks) > 1:
        block_arr = np.array(blocks)
        noise_uniformity = 1.0 - float(np.clip(block_arr.std() / (block_arr.mean() + 1e-8), 0, 1))
    else:
        noise_uniformity = 0.5

    return float(np.clip(0.5 * low_noise_score + 0.5 * noise_uniformity, 0, 1))


# ─── signal 4: texture uniformity ──────────────────────────────────────────────

def _texture_score(img: np.ndarray) -> float:
    """
    AI portrait generators produce suspiciously uniform, smooth skin textures.
    Edited images may show boundary artifacts or inconsistent texture regions.
    """
    gray = _to_gray(img)
    h, w = gray.shape

    kernel_size = max(min(h, w) // 16, 7) | 1
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    blur_sq = cv2.GaussianBlur(gray**2, (kernel_size, kernel_size), 0)
    local_var = np.sqrt(np.abs(blur_sq - blur**2))

    lv_of_lv = local_var.std() / (local_var.mean() + 1e-8)
    uniformity_score = float(np.clip(1.0 - lv_of_lv / 1.5, 0, 1))

    hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [64], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-8)
    hist_roughness = float(np.std(np.diff(hist)) * 50)
    smooth_hist_score = float(np.clip(1.0 - hist_roughness, 0, 1))

    return float(np.clip(0.6 * uniformity_score + 0.4 * smooth_hist_score, 0, 1))


# ─── signal 5: gradient/edge proxy ─────────────────────────────────────────────

def _edge_score(img: np.ndarray) -> float:
    """
    AI images from diffusion models have abnormally sharp, consistent edges.
    Edited images may have inconsistent edge sharpness at manipulation boundaries.
    """
    gray = _to_gray(img)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx**2 + gy**2)

    flat = gmag.flatten()
    p25, p75 = np.percentile(flat, 25), np.percentile(flat, 75)
    iqr = p75 - p25 + 1e-8
    median_grad = np.median(flat) + 1e-8
    uniformity = float(np.clip(1.0 - iqr / (median_grad * 2), 0, 1))

    r, g, b = img[:, :, 0].flatten(), img[:, :, 1].flatten(), img[:, :, 2].flatten()
    rg_corr = abs(float(np.corrcoef(r, g)[0, 1]))
    rb_corr = abs(float(np.corrcoef(r, b)[0, 1]))
    channel_corr = float(np.clip((rg_corr + rb_corr) / 2, 0, 1))

    return float(np.clip(0.5 * uniformity + 0.5 * channel_corr, 0, 1))


# ─── heatmap ───────────────────────────────────────────────────────────────────

def _generate_heatmap(img: np.ndarray) -> str:
    """Noise residual heatmap: highlights regions with anomalous local statistics."""
    gray = _to_gray(img).astype(np.uint8)

    laplacian = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F))
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    texture_diff = cv2.absdiff(gray, blur).astype(np.float32)

    combined = 0.5 * laplacian + 0.5 * texture_diff
    combined = cv2.GaussianBlur(combined, (21, 21), 0)
    combined_norm = (combined / (combined.max() + 1e-8) * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(combined_norm, cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 0.45, heatmap, 0.55, 0)

    filename = f"heatmap_{uuid.uuid4().hex[:8]}.jpg"
    cv2.imwrite(str(STATIC_DIR / filename), overlay)
    return f"/static/{filename}"


# ─── trained model inference ────────────────────────────────────────────────────

def _cnn_predict(img: np.ndarray) -> tuple[str, dict[str, float]]:
    """Run trained EfficientNet-B4 and return (class_name, {class: prob})."""
    import torch
    pil_img = Image.fromarray(img)
    tensor = _TRANSFORM(pil_img).unsqueeze(0)
    probs = _MODEL.predict_proba(tensor).squeeze(0).tolist()
    class_probs = {CLASSES[i]: round(probs[i], 4) for i in range(len(CLASSES))}
    pred_class = CLASSES[int(np.argmax(probs))]
    return pred_class, class_probs


# ─── heuristic 3-class prediction ──────────────────────────────────────────────

def _heuristic_predict(
    spectral: float, gan: float, noise: float, texture: float, edge: float
) -> tuple[str, dict[str, float], float]:
    """
    Returns (prediction, class_probabilities, composite_fake_score).

    Heuristic logic:
      AI_GENERATED: strong GAN fingerprint + spectral deviation
      EDITED:       noise inconsistency + texture anomaly, but weaker GAN signal
    """
    composite = (
        0.28 * spectral
        + 0.22 * gan
        + 0.25 * noise
        + 0.15 * texture
        + 0.10 * edge
    )

    threshold = 0.45
    if composite < threshold:
        real_p = 1.0 - composite
        fake_p = composite
        return "REAL", {
            "REAL": round(real_p, 4),
            "AI_GENERATED": round(fake_p * 0.5, 4),
            "EDITED": round(fake_p * 0.5, 4),
        }, composite

    # Fake — decide subtype
    # AI_GENERATED: GAN peaks + spectral are the dominant signal
    # EDITED: noise inconsistency / texture boundary effects dominate
    ai_score   = 0.50 * gan + 0.35 * spectral + 0.15 * edge
    edit_score = 0.45 * noise + 0.35 * texture + 0.20 * spectral

    total = ai_score + edit_score + 1e-8
    ai_frac   = ai_score / total
    edit_frac = edit_score / total

    fake_mass = composite  # how much probability goes to fake classes
    real_p    = 1.0 - fake_mass
    ai_p      = fake_mass * ai_frac
    edit_p    = fake_mass * edit_frac

    if ai_score >= edit_score:
        prediction = "AI_GENERATED"
    else:
        prediction = "EDITED"

    return prediction, {
        "REAL": round(real_p, 4),
        "AI_GENERATED": round(ai_p, 4),
        "EDITED": round(edit_p, 4),
    }, composite


# ─── signal label → dominant forgery type ──────────────────────────────────────

def _dominant_signal(spectral, gan, noise, texture, edge, prediction) -> str:
    if prediction == "REAL":
        return "none"
    signal_map = {
        "spectral_anomaly":   spectral,
        "gan_upsampling":     gan,
        "noise_inconsistency": noise,
        "texture_boundary":   texture,
        "edge_sharpness":     edge,
    }
    return max(signal_map, key=signal_map.get)


# ─── main entry ────────────────────────────────────────────────────────────────

def analyze(image_data: bytes) -> dict:
    img = _load_image(image_data)

    spectral  = _spectral_score(img)
    gan       = _gan_periodic_score(img)
    noise     = _noise_residual_score(img)
    texture   = _texture_score(img)
    edge      = _edge_score(img)

    if _MODEL is not None:
        prediction, class_probs = _cnn_predict(img)
        is_fake = prediction != "REAL"
        confidence = class_probs[prediction]
    else:
        prediction, class_probs, composite = _heuristic_predict(spectral, gan, noise, texture, edge)
        is_fake = prediction != "REAL"
        confidence = class_probs[prediction]

    forgery_type = _dominant_signal(spectral, gan, noise, texture, edge, prediction)
    heatmap_url  = _generate_heatmap(img) if is_fake else None

    return {
        "prediction": prediction,
        "confidence": round(confidence, 3),
        "forgery_type": forgery_type,
        "heatmap_url": heatmap_url,
        "class_probabilities": class_probs,
        "analysis": {
            "cnn_score":     round(edge, 3),
            "fft_score":     round(spectral, 3),
            "gan_score":     round(gan, 3),
            "noise_score":   round(noise, 3),
            "texture_score": round(texture, 3),
        },
    }
