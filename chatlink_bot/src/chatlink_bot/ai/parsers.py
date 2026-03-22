# src/chatlink_bot/ai/parsers.py
import asyncio
import logging
import os
import base64
import tempfile
from io import BytesIO
from typing import Optional

from cudara_client import CudaraClient, CudaraError, GenerationOptions

logger = logging.getLogger("Parsers")

CUDARA_URL = os.getenv("CUDARA_URL", "http://cudara:8000")
VISION_MODEL = os.getenv("CUDARA_VISION_MODEL", "unsloth/Qwen2.5-VL-3B-Instruct-GGUF")
ASR_MODEL = os.getenv("CUDARA_ASR_MODEL", "openai/whisper-small")

_client: Optional[CudaraClient] = None


def _get_client() -> CudaraClient:
    global _client
    if _client is None:
        _client = CudaraClient(CUDARA_URL)
    return _client


def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    if not image_bytes:
        return ""

    client = _get_client()
    try:
        # Encode to base64 string for the new client payload
        b64_img = base64.b64encode(image_bytes).decode('utf-8')
        resp = client.generate(
            model=VISION_MODEL,
            prompt="Extrae todo el texto de esta imagen. Si no hay texto, descríbeme lo que ves.",
            images=[b64_img],
        )
        return (resp.content or "").strip()
    except CudaraError as e:
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
            
            # Removed generate_kwargs; the new client handles ASR options natively
            resp = client.transcribe(
                model=ASR_MODEL, 
                audio_path=f.name
            )
            # Response text is now stored in .content
            return (resp.content or "").strip()
    except CudaraError as e:
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
