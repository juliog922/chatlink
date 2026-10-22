# chatlink_bot/src/chatlink_bot/logic/alerts.py
"""Monitoring, all of it: alert() emails a problem, heartbeat() pings an
external service that emails you when the pings stop (box down, loop stuck,
SMTP broken).

    ALERT_EMAILS=you@example.com,other@example.com
    HEARTBEAT_URL=https://hc-ping.com/<uuid>     (healthchecks.io or similar)
"""
import logging
import os
import threading
import time
import urllib.request
from typing import Dict, Iterable

logger = logging.getLogger("Alerts")

ALERT_EMAILS = [a.strip() for a in os.getenv("ALERT_EMAILS", "").split(",") if a.strip()]
HEARTBEAT_URL = os.getenv("HEARTBEAT_URL", "").strip()
REPEAT_S = int(os.getenv("ALERT_REPEAT_S", "3600"))

_last: Dict[str, float] = {}
_lock = threading.Lock()


def alert(key: str, text: str, also_to: Iterable[str] = ()) -> bool:
    """Email `text` to the admins (and `also_to`), at most once per REPEAT_S
    for the same `key`. Never raises, never blocks. True if it was sent."""
    now = time.monotonic()
    with _lock:
        if now - _last.get(key, -REPEAT_S) < REPEAT_S:
            return False
        _last[key] = now
    logger.error(f"[ALERT] {key}: {text}")
    to = list(dict.fromkeys(ALERT_EMAILS + [a for a in also_to if a]))
    if to:
        threading.Thread(target=_email, args=(to, key, text), daemon=True).start()
    return True


def _email(to, key, text) -> None:
    from ..transport.email import email_transport
    for addr in to:
        ok, err = email_transport.send_email(to_email=addr, subject=f"[Kapa] {key}", body=text)
        if not ok:
            logger.error(f"[ALERT] could not email {addr}: {err}")


def heartbeat(ok: bool = True) -> None:
    """Ping HEARTBEAT_URL (its /fail variant when not ok). Never raises."""
    if not HEARTBEAT_URL:
        return
    url = HEARTBEAT_URL if ok else HEARTBEAT_URL.rstrip("/") + "/fail"

    def ping():
        try:
            urllib.request.urlopen(url, timeout=10).read()
        except Exception as e:
            logger.warning(f"heartbeat failed: {e}")
    threading.Thread(target=ping, daemon=True).start()
