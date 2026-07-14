# src/chatlink_bot/ai/cima_client.py
"""
Minimal, dependency-free HTTP client for the cima inference engine.

Replaces the old `cudara_client` package: everything here is stdlib
(urllib + json), so there is nothing to install. cima speaks the Ollama
REST API, so the surface mirrors what the bot already used —
``chat`` / ``embed`` / ``tags`` / ``pull`` — plus an ``audio`` helper that
routes through ``/api/chat`` (gemma-4 handles audio natively; there is no
separate transcription endpoint).

One model does everything (text + vision + audio + embedding), so callers
pass a single model name; see ``CIMA_MODEL`` in the modules that use this.

Design notes:
* Blocking ``urllib`` calls — callers wrap in ``asyncio.to_thread`` exactly
  as they did with the old client.
* ``chat``/``generate`` are non-streaming (``stream=False``): the bot always
  consumed the full reply, and non-stream gives byte-exact stop-trimming.
* Reranking is NOT supported by cima (``/api/embed`` returns vectors, never
  scores); ``rerank`` returns empty so the RAG layer falls back to its RRF
  path. This is intentional and documented, not a silent stub.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("CimaClient")

DEFAULT_TIMEOUT = 300.0  # generous: generation on a 6 GiB card can be slow


@dataclass
class Message:
    """A single chat turn. ``images``/``audio`` are lists of base64 strings."""
    role: str
    content: str = ""
    images: Optional[List[str]] = None
    audio: Optional[List[str]] = None

    def to_json(self) -> Dict[str, Any]:
        m: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.images:
            m["images"] = list(self.images)
        if self.audio:
            m["audio"] = list(self.audio)
        return m


@dataclass
class ChatResponse:
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    done_reason: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbedResponse:
    embeddings: List[List[float]] = field(default_factory=list)
    # cima never returns rerank scores; kept for call-site compatibility.
    scores: List[float] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


class CimaError(RuntimeError):
    """Raised when cima returns a non-2xx response, carrying its message."""

    def __init__(self, status: int, message: str):
        super().__init__(f"cima HTTP {status}: {message}")
        self.status = status
        self.message = message


class CimaClient:
    def __init__(self, base_url: str = "http://cima:8000", timeout: float = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ---- low-level -------------------------------------------------------

    def _post(self, path: str, payload: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            try:
                detail = json.loads(detail).get("error", detail)
            except Exception:
                pass
            raise CimaError(e.code, detail) from None
        return json.loads(body) if body else {}

    def _get(self, path: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        req = urllib.request.Request(f"{self.base_url}{path}", method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            raise CimaError(e.code, detail) from None
        return json.loads(body) if body else {}

    # ---- chat / generate -------------------------------------------------

    def chat(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        options: Optional[Dict[str, Any]] = None,
        fmt: Optional[str] = None,          # "json" or a JSON-schema dict
        tools: Optional[List[Dict[str, Any]]] = None,
        timeout: Optional[float] = None,
    ) -> ChatResponse:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [m.to_json() for m in messages],
            "stream": False,
        }
        if options:
            payload["options"] = options
        if fmt is not None:
            payload["format"] = fmt
        if tools:
            payload["tools"] = tools

        data = self._post("/api/chat", payload, timeout=timeout)
        msg = data.get("message") or {}
        return ChatResponse(
            content=msg.get("content") or "",
            tool_calls=msg.get("tool_calls") or [],
            done_reason=data.get("done_reason") or "",
            raw=data,
        )

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        options: Optional[Dict[str, Any]] = None,
        fmt: Optional[str] = None,
        images: Optional[List[str]] = None,
        audio: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> ChatResponse:
        payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        if options:
            payload["options"] = options
        if fmt is not None:
            payload["format"] = fmt
        if images:
            payload["images"] = images
        if audio:
            payload["audio"] = audio
        data = self._post("/api/generate", payload, timeout=timeout)
        return ChatResponse(
            content=data.get("response") or "",
            done_reason=data.get("done_reason") or "",
            raw=data,
        )

    # ---- embeddings ------------------------------------------------------

    def embed(
        self,
        model: str,
        input: Any,                          # str | list[str]  (name kept for compat)
        *,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> EmbedResponse:
        # cima has no rerank; if a caller asks for it, return empty so the RAG
        # layer falls back to RRF rather than believing it got scores.
        if options and options.get("is_rerank"):
            logger.debug("cima does not support reranking; returning empty scores.")
            return EmbedResponse(embeddings=[], scores=[], raw={})
        payload: Dict[str, Any] = {"model": model, "input": input}
        if options:
            payload["options"] = options
        data = self._post("/api/embed", payload, timeout=timeout)
        return EmbedResponse(embeddings=data.get("embeddings") or [], raw=data)

    # ---- model management ------------------------------------------------

    def tags(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        return self._get("/api/tags", timeout=timeout)

    def pull(self, model: str, stream: bool = False, timeout: Optional[float] = None) -> Dict[str, Any]:
        # cima pulls server-side; a short client timeout is expected to fire
        # while the download continues in the background (callers handle it).
        return self._post("/api/pull", {"model": model, "stream": stream}, timeout=timeout)

    def ready(self, models: Optional[List[str]] = None, timeout: Optional[float] = None) -> Dict[str, Any]:
        if models:
            return self._post("/api/ready", {"models": models}, timeout=timeout)
        return self._get("/api/ready", timeout=timeout)

    def version(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        return self._get("/api/version", timeout=timeout)