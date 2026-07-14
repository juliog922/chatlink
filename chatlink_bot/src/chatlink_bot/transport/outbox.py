# chatlink_bot/src/chatlink_bot/transport/outbox.py
"""
Two tiny in-memory registries (stdlib only) shared by handlers and transports.

RecentOutbox — fixes the bot-echo race. The bot sends from the salesman's own
WhatsApp JID / Gmail account, so every outbound bot message echoes back on the
inbound stream as a "sent" message. The old defence queried Postgres for an
identical persisted bot row, but the echo can arrive BEFORE the commit; when
it did, the echo was classified as a manual salesman reply and triggered an
FSM handoff — the bot cancelled itself. Registering the fingerprint *before*
sending makes the check ordering-safe by construction.

SimCapture — records the bot's outbound replies for simulated conversations
so the test API can return them synchronously instead of forcing the caller
to poll the database.
"""
from __future__ import annotations

import hashlib
import re
import time
from collections import deque
from typing import Deque, Dict, List, Tuple, Any, Optional

Key = Tuple[str, str, str]  # (channel, client_id, user_id)


def _digest(text: str) -> str:
    # Whitespace-normalized: Gmail returns the bot's own sent mail with CRLF
    # line endings and trailing newlines, which must still match the original.
    normalized = re.sub(r"\s+", " ", text or "").strip()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def same_text(a: str, b: str) -> bool:
    """Whitespace-insensitive equality — the trigger's last-resort loop breaker."""
    return _digest(a) == _digest(b)


class RecentOutbox:
    """Short-lived fingerprints of texts the bot just sent, per conversation."""

    def __init__(self, ttl_s: float = 300.0, max_per_key: int = 30) -> None:
        self.ttl_s = ttl_s
        self._sent: Dict[Key, Deque[Tuple[str, float]]] = {}
        self._max_per_key = max_per_key

    def register(self, key: Key, text: str, copies: int = 1) -> None:
        """Call BEFORE handing the text to the transport. `copies` > 1 for
        SELF-conversations (salesman chatting with their own number/mailbox),
        where one send can echo back through several paths — WhatsApp device
        sync, and for email both the inbox AND the sent-folder poller."""
        bucket = self._sent.setdefault(key, deque(maxlen=self._max_per_key))
        now = time.monotonic()
        for _ in range(max(1, copies)):
            bucket.append((_digest(text), now))

    def is_echo(self, key: Key, text: str) -> bool:
        """True if `text` matches a recent bot send for this conversation (consumes the match)."""
        bucket = self._sent.get(key)
        if not bucket:
            return False
        now, wanted = time.monotonic(), _digest(text)
        fresh = [(d, ts) for d, ts in bucket if now - ts <= self.ttl_s]
        hit = any(d == wanted for d, _ in fresh)
        if hit:  # consume one occurrence so N sends tolerate exactly N echoes
            for i, (d, _) in enumerate(fresh):
                if d == wanted:
                    fresh.pop(i)
                    break
        self._sent[key] = deque(fresh, maxlen=self._max_per_key)
        return hit


class SimCapture:
    """Outbound bot replies for simulated conversations, drained by the test API."""

    def __init__(self, max_per_key: int = 50) -> None:
        self._replies: Dict[Key, List[Dict[str, Any]]] = {}
        self._max_per_key = max_per_key

    def record(self, key: Key, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self._replies.setdefault(key, [])
        self._replies[key] = self._replies[key][-self._max_per_key:] + [{"text": text, "meta": meta or {}}]

    def drain(self, key: Key) -> List[Dict[str, Any]]:
        """Return and clear captured replies (dicts: text + meta) for a conversation."""
        return self._replies.pop(key, [])


bot_outbox = RecentOutbox()
sim_capture = SimCapture()