# src/chatlink_bot/ai/parsers.py
import asyncio
import logging
import os
import base64
import tempfile
from io import BytesIO
from typing import Optional

from PIL import Image, ImageEnhance, ImageFilter, ExifTags

from cudara_client import CudaraClient, Message

logger = logging.getLogger("Parsers")

CUDARA_URL = os.getenv("CUDARA_URL", "http://cudara:8000")
VISION_MODEL = os.getenv("CUDARA_VISION_MODEL", "unsloth/Qwen2.5-VL-3B-Instruct-GGUF")
ASR_MODEL = os.getenv("CUDARA_ASR_MODEL", "openai/whisper-small")

# ---------------------------------------------------------------------------
# Image preprocessing tunables (override via env for quick experiments)
# ---------------------------------------------------------------------------
# Minimum dimension before upscaling kicks in.  WhatsApp thumbnails and
# heavily-compressed photos often land at 800-1200px on the longest side;
# Qwen2.5-VL performs noticeably better above ~1500px.
IMG_MIN_LONG_EDGE = int(os.getenv("IMG_MIN_LONG_EDGE", "1500"))

# Upper pixel-count cap BEFORE sending to Cudara.  Keeps base64 payload and
# server-side tensor allocation sane.  30 MP matches the Cudara server default.
IMG_MAX_PIXELS = int(os.getenv("IMG_MAX_PIXELS", str(30_000_000)))

# Sharpening strength applied via UnsharpMask to counteract JPEG blur.
# 0 = disabled.  Good range for WhatsApp photos: 1.0-2.0
IMG_SHARPEN_RADIUS = float(os.getenv("IMG_SHARPEN_RADIUS", "2"))
IMG_SHARPEN_PERCENT = int(os.getenv("IMG_SHARPEN_PERCENT", "150"))
IMG_SHARPEN_THRESHOLD = int(os.getenv("IMG_SHARPEN_THRESHOLD", "3"))

# Mild contrast boost (1.0 = no change, 1.1-1.3 helps washed-out photos)
IMG_CONTRAST_FACTOR = float(os.getenv("IMG_CONTRAST_FACTOR", "1.15"))

# Mild sharpness boost via Pillow's Sharpness enhancer (stacks with USM)
IMG_SHARPNESS_FACTOR = float(os.getenv("IMG_SHARPNESS_FACTOR", "1.3"))

_client: Optional[CudaraClient] = None


def _get_client() -> CudaraClient:
    global _client
    if _client is None:
        _client = CudaraClient(CUDARA_URL)
    return _client


# ---------------------------------------------------------------------------
# Image preprocessing pipeline for WhatsApp-compressed photos
# ---------------------------------------------------------------------------

def _fix_exif_orientation(img: Image.Image) -> Image.Image:
    """Apply EXIF orientation transforms so text isn't sideways."""
    try:
        exif = img.getexif()
        orientation_key = next(
            (tag_id for tag_id, name in ExifTags.TAGS.items() if name == "Orientation"),
            None,
        )
        if orientation_key and orientation_key in exif:
            _T = Image.Transpose
            transforms = {
                2: [_T.FLIP_LEFT_RIGHT],
                3: [_T.ROTATE_180],
                4: [_T.FLIP_TOP_BOTTOM],
                5: [_T.FLIP_LEFT_RIGHT, _T.ROTATE_90],
                6: [_T.ROTATE_270],
                7: [_T.FLIP_LEFT_RIGHT, _T.ROTATE_270],
                8: [_T.ROTATE_90],
            }
            for op in transforms.get(exif[orientation_key], []):
                img = img.transpose(op)
    except Exception:
        pass
    return img


def _ensure_rgb(img: Image.Image) -> Image.Image:
    """Convert any mode (RGBA, P, LA, L, CMYK …) to clean RGB."""
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA", "PA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        # Use alpha channel as mask if present
        if "A" in img.mode:
            rgba = img.convert("RGBA")
            background.paste(rgba, mask=rgba.split()[3])
        else:
            background.paste(img)
        return background
    if img.mode == "P":
        return _ensure_rgb(img.convert("RGBA"))
    # L, CMYK, etc.
    return img.convert("RGB")


def _preprocess_image(raw_bytes: bytes) -> bytes:
    """
    Full preprocessing pipeline for WhatsApp images before VLM OCR:
    
    1. Decode & validate
    2. Fix EXIF orientation (WhatsApp/phone rotation)
    3. Convert to RGB
    4. Upscale small images (Qwen2.5-VL needs sufficient resolution)
    5. Apply UnsharpMask (counteract JPEG compression blur)
    6. Mild contrast & sharpness boost
    7. Encode as lossless PNG (avoid re-compression artifacts)
    
    Returns PNG bytes ready for base64 encoding.
    """
    img = Image.open(BytesIO(raw_bytes))
    img.load()  # Force full decode — catches truncated files early

    # --- EXIF orientation (critical for phone photos) ---
    img = _fix_exif_orientation(img)

    # --- RGB conversion ---
    img = _ensure_rgb(img)

    w, h = img.size

    # --- Downscale if above pixel cap (safety valve) ---
    if w * h > IMG_MAX_PIXELS:
        scale = (IMG_MAX_PIXELS / (w * h)) ** 0.5
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        w, h = new_w, new_h
        logger.debug("Downscaled to %dx%d (pixel cap)", w, h)

    # --- Upscale small images ---
    # WhatsApp often delivers ~800-1200px.  Qwen2.5-VL's dynamic resolution
    # slicing performs better when text characters span more pixels.
    long_edge = max(w, h)
    if long_edge < IMG_MIN_LONG_EDGE and long_edge > 0:
        scale = IMG_MIN_LONG_EDGE / long_edge
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        logger.debug("Upscaled from %dx%d to %dx%d", w, h, new_w, new_h)

    # --- UnsharpMask (counteracts JPEG blur, the #1 OCR killer) ---
    if IMG_SHARPEN_RADIUS > 0:
        img = img.filter(
            ImageFilter.UnsharpMask(
                radius=IMG_SHARPEN_RADIUS,
                percent=IMG_SHARPEN_PERCENT,
                threshold=IMG_SHARPEN_THRESHOLD,
            )
        )

    # --- Contrast boost (helps washed-out/low-light photos) ---
    if IMG_CONTRAST_FACTOR != 1.0:
        img = ImageEnhance.Contrast(img).enhance(IMG_CONTRAST_FACTOR)

    # --- Sharpness boost (stacks with USM for fine text edges) ---
    if IMG_SHARPNESS_FACTOR != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(IMG_SHARPNESS_FACTOR)

    # --- Encode as lossless PNG ---
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Qwen 2.5-VL optimized OCR prompt
# ---------------------------------------------------------------------------

_OCR_SYSTEM_PROMPT = (
    "Eres un motor de OCR de alta precisión. Tu ÚNICA tarea es extraer fielmente "
    "todo el texto visible en la imagen, preservando la estructura original "
    "(saltos de línea, listas, tablas, columnas). "
    "Reglas estrictas:\n"
    "- Transcribe EXACTAMENTE lo que ves, sin corregir ortografía ni gramática.\n"
    "- Mantén el idioma original del texto.\n"
    "- Para tablas, usa tabuladores o alineación con espacios.\n"
    "- Si hay texto parcialmente ilegible, indica las partes dudosas entre corchetes: [ilegible].\n"
    "- Si la imagen NO contiene texto, responde ÚNICAMENTE con una descripción "
    "breve y objetiva de lo que muestra la imagen (máximo 2 oraciones)."
)

_OCR_USER_PROMPT = (
    "Extrae todo el texto visible en esta imagen. "
    "Preserva la estructura, el orden de lectura y el formato original."
)


def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    """
    Full pipeline: preprocess WhatsApp-compressed image → send to Qwen2.5-VL.
    
    Improvements over raw passthrough:
    - EXIF orientation fix (sideways phone photos)
    - Upscale small images for better VLM resolution slicing
    - UnsharpMask + contrast to counteract JPEG compression artifacts
    - Lossless PNG encoding (no re-compression)
    - Structured system + user prompt tuned for Qwen2.5-VL OCR
    - temperature=0 for deterministic, faithful extraction
    """
    if not image_bytes:
        return ""

    client = _get_client()
    try:
        # --- Preprocess: enhance the WhatsApp-compressed image ---
        try:
            clean_png = _preprocess_image(image_bytes)
        except Exception as e:
            # If preprocessing fails (truly corrupt image), fall back to raw bytes
            logger.warning(f"Image preprocessing failed, sending raw: {e}")
            clean_png = image_bytes

        b64_img = base64.b64encode(clean_png).decode("utf-8")

        resp = client.chat(
            model=VISION_MODEL,
            messages=[
                Message(role="system", content=_OCR_SYSTEM_PROMPT),
                Message(
                    role="user",
                    content=_OCR_USER_PROMPT,
                    images=[b64_img],
                ),
            ],
            options={
                "temperature": 0.0,   # Deterministic — faithful to the image
                "top_p": 1.0,         # No nucleus sampling for OCR
                "repeat_penalty": 1.0, # Don't penalize repeated text (tables, lists)
                "num_predict": 2048,   # Allow long extractions (invoices, documents)
            },
        )
        return (resp.content or "").strip()
    except Exception as e:
        logger.error(f"Vision OCR failed: {e}")
        return ""


def transcribe_audio_bytes(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    if not audio_bytes:
        return ""

    client = _get_client()
    suffix = ".wav"
    if "." in (filename or ""):
        suffix = "." + filename.rsplit(".", 1)[-1].lower()

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as f:
            f.write(audio_bytes)
            f.flush()

            resp = client.transcribe(
                model=ASR_MODEL,
                audio_path=f.name,
            )
            return (resp.content or "").strip()
    except Exception as e:
        logger.error(f"Audio transcription failed: {e}")
        return ""


def extract_text_from_document_bytes(data: bytes, filename: str) -> str:
    """
    Best-effort doc extraction for: txt/csv/json/md, pdf, docx, xlsx.
    """
    if not data:
        return ""

    fn = (filename or "").lower().strip()

    # text-like
    if fn.endswith((".txt", ".csv", ".md", ".json")):
        for enc in ("utf-8", "latin-1"):
            try:
                return data.decode(enc, errors="strict").strip()
            except Exception:
                continue
        return data.decode("utf-8", errors="replace").strip()

    # pdf
    if fn.endswith(".pdf"):
        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(BytesIO(data))
            out = []
            for page in reader.pages:
                out.append(page.extract_text() or "")
            return "\n".join([x.strip() for x in out if x.strip()]).strip()
        except Exception as e:
            logger.warning(f"PDF extract failed: {e}")
            return ""

    # docx
    if fn.endswith(".docx"):
        try:
            from docx import Document  # type: ignore

            with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as f:
                f.write(data)
                f.flush()
                doc = Document(f.name)
            paras = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
            return "\n".join(paras).strip()
        except Exception as e:
            logger.warning(f"DOCX extract failed: {e}")
            return ""

    # xlsx/xlsm/xltx...
    if fn.endswith((".xlsx", ".xlsm", ".xltx", ".xltm", ".xls")):
        try:
            from openpyxl import load_workbook  # type: ignore

            wb = load_workbook(BytesIO(data), read_only=True, data_only=True)
            out: list[str] = []
            for ws in wb.worksheets:
                for row in ws.iter_rows(values_only=True):
                    vals = [str(v).strip() for v in row if v is not None and str(v).strip()]
                    if vals:
                        out.append(" ".join(vals))
            return "\n".join(out).strip()
        except Exception as e:
            logger.warning(f"XLSX extract failed: {e}")
            return ""

    return ""


async def extract_text_from_image_bytes_async(image_bytes: bytes) -> str:
    return await asyncio.to_thread(extract_text_from_image_bytes, image_bytes)


async def transcribe_audio_bytes_async(audio_bytes: bytes, filename: str) -> str:
    return await asyncio.to_thread(transcribe_audio_bytes, audio_bytes, filename)


async def extract_text_from_document_bytes_async(data: bytes, filename: str) -> str:
    return await asyncio.to_thread(extract_text_from_document_bytes, data, filename)