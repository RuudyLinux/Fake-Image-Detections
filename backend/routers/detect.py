from fastapi import APIRouter, UploadFile, File, HTTPException
from models import blind_detector, comparative_detector

router = APIRouter(prefix="/detect", tags=["detection"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_SIZE = 20 * 1024 * 1024  # 20 MB


def _validate_image(upload: UploadFile) -> None:
    if upload.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"Unsupported image type: {upload.content_type}")


@router.post("/blind")
async def detect_blind(image: UploadFile = File(...)):
    _validate_image(image)
    data = await image.read()
    if len(data) > MAX_SIZE:
        raise HTTPException(413, "Image exceeds 20 MB limit")
    try:
        result = blind_detector.analyze(data)
    except Exception as exc:
        raise HTTPException(500, f"Analysis error: {exc}") from exc
    return result


@router.post("/compare")
async def detect_comparative(
    original: UploadFile = File(...),
    suspected: UploadFile = File(...),
):
    _validate_image(original)
    _validate_image(suspected)
    orig_data = await original.read()
    susp_data = await suspected.read()
    if len(orig_data) > MAX_SIZE or len(susp_data) > MAX_SIZE:
        raise HTTPException(413, "Image exceeds 20 MB limit")
    try:
        result = comparative_detector.compare(orig_data, susp_data)
    except Exception as exc:
        raise HTTPException(500, f"Comparison error: {exc}") from exc
    return result
