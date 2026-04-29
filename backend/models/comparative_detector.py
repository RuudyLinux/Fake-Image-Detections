"""
Comparative detection: SSIM + pixel diff + contour-based region localization.
"""

import io
import uuid
import numpy as np
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
from pathlib import Path

STATIC_DIR = Path(__file__).parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

MAX_DIM = 1024


def _load_resize(data: bytes, target_size: tuple | None = None) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if target_size:
        img = img.resize(target_size, Image.LANCZOS)
    else:
        w, h = img.size
        scale = min(MAX_DIM / w, MAX_DIM / h, 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img)


def _save_image(img: np.ndarray, prefix: str) -> str:
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.jpg"
    out_path = STATIC_DIR / filename
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return f"/static/{filename}"


def compare(original_data: bytes, suspected_data: bytes) -> dict:
    orig = _load_resize(original_data)
    h, w = orig.shape[:2]
    susp = _load_resize(suspected_data, target_size=(w, h))

    orig_gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    susp_gray = cv2.cvtColor(susp, cv2.COLOR_RGB2GRAY)

    # SSIM
    score, ssim_map = ssim(orig_gray, susp_gray, full=True)
    score = float(score)

    # Pixel diff map
    diff = cv2.absdiff(orig, susp)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    diff_norm = (diff_gray.astype(np.float32) / (diff_gray.max() + 1e-8) * 255).astype(np.uint8)
    diff_colored = cv2.applyColorMap(diff_norm, cv2.COLORMAP_HOT)
    diff_map_url = _save_image(cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB), "diff")

    # Highlight manipulated regions on suspected image
    _, thresh = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    significant = [c for c in contours if cv2.contourArea(c) > 200]
    highlighted = susp.copy()
    cv2.drawContours(highlighted, significant, -1, (255, 50, 50), 2)
    for c in significant:
        x, y, rw, rh = cv2.boundingRect(c)
        cv2.rectangle(highlighted, (x, y), (x + rw, y + rh), (255, 80, 80), 1)

    highlighted_url = _save_image(highlighted, "highlighted") if significant else None
    is_manipulated = score < 0.95 or len(significant) > 0

    return {
        "similarity_score": round(score, 4),
        "is_manipulated": bool(is_manipulated),
        "diff_map_url": diff_map_url,
        "highlighted_url": highlighted_url,
        "regions_count": len(significant),
    }
