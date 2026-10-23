# chatlink_bot/src/chatlink_bot/ai/inference.py
"""Tachyon, through its command line.

With `tachyon serve` running (the tachyon container, socket shared through
TACHYON_SOCKET), `run`, `transcribe` and `decide` go through the daemon and
models stay in VRAM, so one subprocess per call costs milliseconds and no
client library has to be written.

    KAPA_CHAT_MODEL   gemma-4-e4b on the server, gemma-4-e2b on a small GPU
    KAPA_LAYA_MODEL   laya-multilingual
"""
import json
import logging
import os
import re
import subprocess
import tempfile
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger("Inference")

TACHYON = os.getenv("TACHYON_BIN", "tachyon")
CHAT_MODEL = os.getenv("KAPA_CHAT_MODEL", "gemma-4-e4b")
LAYA_MODEL = os.getenv("KAPA_LAYA_MODEL", "laya-multilingual")
TIMEOUT_S = float(os.getenv("INFERENCE_TIMEOUT_S", "120"))
# A Laya pick below this probability becomes a question instead.
LAYA_PICK = float(os.getenv("LAYA_PICK", "0.6"))


class InferenceError(RuntimeError):
    pass


def _tachyon(args: Sequence[str], timeout: float = TIMEOUT_S) -> str:
    try:
        p = subprocess.run([TACHYON, *args], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise InferenceError(f"tachyon {args[0]}: {e}") from e
    if p.returncode != 0:
        raise InferenceError(f"tachyon {args[0]} exited {p.returncode}: {p.stderr.strip()[-300:]}")
    return p.stdout


def _tmp(data: bytes, suffix: str) -> str:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(data)
    f.close()
    return f.name


def chat(prompt: str, images: Sequence[bytes] = (), timeout: float = TIMEOUT_S) -> str:
    """Gemma answers `prompt` greedily, shown the PNG/JPEG `images` first."""
    files = [_tmp(b, ".png") for b in images]
    try:
        return _tachyon(["run", CHAT_MODEL, prompt, *[f"--image={f}" for f in files]], timeout).strip()
    finally:
        for f in files:
            os.unlink(f)


def transcribe(wav: bytes, timeout: float = TIMEOUT_S) -> str:
    """Speech in a WAV clip (up to 30 s) as text."""
    f = _tmp(wav, ".wav")
    try:
        return _tachyon(["transcribe", CHAT_MODEL, f], timeout).strip()
    finally:
        os.unlink(f)


def decide(state: str, questions: Dict[str, dict], timeout: float = 30) -> Dict[str, dict]:
    """Laya: {name: {"type": "choice", "instructions": ..., "criteria": {label: text}}}
    -> {name: {"choice": label, "probabilities": {label: p}}}."""
    f = _tmp(json.dumps({"state": state, "questions": questions}, ensure_ascii=False).encode(), ".json")
    try:
        return json.loads(_tachyon(["decide", LAYA_MODEL, f], timeout))
    finally:
        os.unlink(f)


def choose(state: str, items: List[dict]) -> List[dict]:
    """For each {"texto", "opciones": [names]}: which option the client means.
    -> [{"item": i, "veredicto": "unico"|"varios"|"ninguno", "indice": int|None, "p": float}]"""
    questions = {}
    for i, it in enumerate(items):
        criteria = {str(j): name for j, name in enumerate(it["opciones"])}
        criteria["ninguno"] = "ninguno de estos productos"
        questions[f"q{i}"] = {"type": "choice", "criteria": criteria,
                              "instructions": f'¿Qué producto pide el cliente con "{it["texto"]}"?'}
    answers = decide(state, questions)
    out = []
    for i in range(len(items)):
        a = answers.get(f"q{i}") or {}
        best = str(a.get("choice", ""))
        p = float((a.get("probabilities") or {}).get(best, 0.0))
        # "ninguno" is only a place for Laya's doubt to go: the candidates come
        # from the client's own history and cover his words, so when Laya
        # rejects them all Kapa asks which one; it never answers "not found".
        if p < LAYA_PICK or best == "ninguno":
            out.append({"item": i, "veredicto": "varios", "indice": None, "p": p})
        else:
            out.append({"item": i, "veredicto": "unico", "indice": int(best), "p": p})
    logger.info("[LAYA] " + ", ".join(f"{it['texto']!r}->{o['veredicto']}({o['p']:.2f})" for it, o in zip(items, out)))
    return out


def json_from(text: str) -> Optional[dict]:
    """The first JSON object in a model answer (tolerates ``` fences and chatter)."""
    text = re.sub(r"```(?:json)?", "", text or "")
    start = text.find("{")
    while start != -1:
        depth = 0
        for k in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            if depth == 0:
                try:
                    data = json.loads(text[start:k + 1])
                    return data if isinstance(data, dict) else None
                except ValueError:
                    break
        start = text.find("{", start + 1)
    return None


def healthy() -> bool:
    try:
        _tachyon(["ps"], timeout=10)
        return True
    except InferenceError:
        return False
