# chatlink_bot/src/chatlink_bot/logic/fsm.py
"""
Per-conversation debounce state machine.

IDLE -> (client message) -> DEBOUNCING -> (timer fires) -> PROCESSING -> IDLE
A manual salesman reply at any point cancels everything (human handoff).

Changes vs the previous version:
- The FSM is the ONLY concurrency authority: `trigger_ai_processing` is never
  emitted twice concurrently for a key (PROCESSING blocks re-entry, new client
  messages during PROCESSING set `pending_redebounce`). The duplicate lock
  dict in handlers is gone.
- The per-entry watchdog task is gone: the periodic janitor already reaps
  conversations stuck in PROCESSING, so one mechanism does that job.
- `on_ai_done` / reaping PEEK at entries instead of creating them (the old
  create-on-read leaked ghost entries).
- `flush()` fires a pending debounce immediately and awaits the full AI
  pipeline — this is what makes the simulator synchronous.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional, Tuple

from ..events import event_bus

logger = logging.getLogger("FSM")

Key = Tuple[str, str, str]  # (channel, client_id, user_id)

DEBOUNCE_MINUTES = float(os.getenv("RESPONSE_DELAY_MINUTES", "15"))
PROCESSING_TIMEOUT_S = int(os.getenv("FSM_PROCESSING_TIMEOUT_S", "300"))
STALE_TTL_S = int(os.getenv("FSM_STALE_TTL_S", "3600"))
CLEANUP_INTERVAL_S = int(os.getenv("FSM_CLEANUP_INTERVAL_S", "120"))
MAX_CONVERSATIONS = int(os.getenv("FSM_MAX_CONVERSATIONS", "5000"))


class ConversationState(str, Enum):
    IDLE = "IDLE"
    DEBOUNCING = "DEBOUNCING"
    PROCESSING = "PROCESSING"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class _Entry:
    state: ConversationState = ConversationState.IDLE
    token: int = 0                                # invalidates stale timers
    debounce_task: Optional[asyncio.Task] = None
    last_activity: datetime = field(default_factory=_now)
    is_simulation: bool = False
    pending_redebounce: bool = False              # client wrote while AI was working
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ConversationFSM:
    def __init__(self, debounce_minutes: float = DEBOUNCE_MINUTES) -> None:
        self.debounce_seconds = debounce_minutes * 60
        self._entries: Dict[Key, _Entry] = {}
        self._map_lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------ map
    async def _entry(self, key: Key, create: bool = True) -> Optional[_Entry]:
        async with self._map_lock:
            entry = self._entries.get(key)
            if entry is None and create:
                entry = self._entries[key] = _Entry()
            return entry

    async def _remove(self, key: Key) -> None:
        async with self._map_lock:
            self._entries.pop(key, None)

    # --------------------------------------------------------------- public
    async def on_client_message(
        self, channel: str, client_id: str, user_id: str, is_simulation: bool = False
    ) -> None:
        """Client wrote: (re)start the inactivity timer, or queue a re-run if AI is busy."""
        key: Key = (channel, client_id, user_id)
        entry = await self._entry(key)
        assert entry is not None
        async with entry.lock:
            entry.last_activity = _now()
            entry.is_simulation = is_simulation
            if entry.state == ConversationState.PROCESSING:
                entry.pending_redebounce = True
                logger.info(f"[FSM] Client msg during PROCESSING for {key}; will re-debounce.")
                return
            entry.state = ConversationState.DEBOUNCING
            self._restart_debounce_locked(key, entry)

    async def on_user_message(self, channel: str, client_id: str, user_id: str) -> None:
        """Salesman replied manually: human handoff — cancel everything and forget."""
        key: Key = (channel, client_id, user_id)
        entry = await self._entry(key, create=False)
        if entry is None:
            return
        async with entry.lock:
            self._cancel_debounce_locked(entry)
            entry.state = ConversationState.IDLE
            entry.pending_redebounce = False
        await self._remove(key)
        logger.info(f"[FSM] Human handoff for {key}.")

    async def on_ai_done(self, channel: str, client_id: str, user_id: str) -> None:
        """AI pipeline finished: go IDLE, or re-debounce if the client wrote meanwhile."""
        key: Key = (channel, client_id, user_id)
        entry = await self._entry(key, create=False)
        if entry is None:
            return
        remove = False
        async with entry.lock:
            entry.last_activity = _now()
            if entry.state != ConversationState.PROCESSING:
                return  # already handed off / reaped
            if entry.pending_redebounce:
                entry.pending_redebounce = False
                entry.state = ConversationState.DEBOUNCING
                self._restart_debounce_locked(key, entry)
                logger.info(f"[FSM] Re-debouncing after AI done for {key}.")
            else:
                entry.state = ConversationState.IDLE
                remove = True
        if remove:
            await self._remove(key)

    async def flush(self, channel: str, client_id: str, user_id: str) -> bool:
        """
        Test/simulation helper: fire a pending debounce NOW and await the whole
        AI pipeline (event handlers are awaited by the bus). True if it fired.
        """
        key: Key = (channel, client_id, user_id)
        entry = await self._entry(key, create=False)
        if entry is None:
            return False
        async with entry.lock:
            if entry.state != ConversationState.DEBOUNCING:
                return False
            self._cancel_debounce_locked(entry)
            token = entry.token
        return await self._fire(key, entry, token)

    # ------------------------------------------------------------- internals
    def _cancel_debounce_locked(self, entry: _Entry) -> None:
        if entry.debounce_task and not entry.debounce_task.done():
            entry.debounce_task.cancel()
        entry.debounce_task = None

    def _restart_debounce_locked(self, key: Key, entry: _Entry) -> None:
        self._cancel_debounce_locked(entry)
        entry.token += 1
        entry.debounce_task = asyncio.create_task(self._debounce_timer(key, entry, entry.token))

    async def _debounce_timer(self, key: Key, entry: _Entry, token: int) -> None:
        try:
            await asyncio.sleep(self.debounce_seconds)
        except asyncio.CancelledError:
            return
        await self._fire(key, entry, token)

    async def _fire(self, key: Key, entry: _Entry, token: int) -> bool:
        """Move DEBOUNCING -> PROCESSING and emit the trigger (outside the lock)."""
        async with entry.lock:
            if entry.token != token or entry.state != ConversationState.DEBOUNCING:
                return False
            entry.state = ConversationState.PROCESSING
            entry.last_activity = _now()
            entry.pending_redebounce = False
            is_simulation = entry.is_simulation
        channel, client_id, user_id = key
        logger.info(f"[{channel}] Timer fired -> PROCESSING client={client_id} user={user_id} (sim={is_simulation})")
        await event_bus.emit(
            "trigger_ai_processing",
            {
                "channel": channel,
                "client_id": client_id,
                "user_id": user_id,
                "fired_at": _now().isoformat(),
                "is_simulation": is_simulation,
            },
        )
        return True

    # -------------------------------------------------------------- janitor
    def start_cleanup_loop(self) -> None:
        """Call once after the event loop is running (lifespan)."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    def stop_cleanup_loop(self) -> None:
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()

    async def _cleanup_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(CLEANUP_INTERVAL_S)
                await self._reap()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"FSM cleanup loop error: {e}")

    async def _reap(self) -> None:
        """Drop stuck PROCESSING entries (crashed handlers), stale IDLE ones, and enforce the cap."""
        now = _now()
        async with self._map_lock:
            doomed: list[Key] = []
            for key, entry in self._entries.items():
                age = (now - entry.last_activity).total_seconds()
                stuck = entry.state == ConversationState.PROCESSING and age > PROCESSING_TIMEOUT_S
                stale = entry.state == ConversationState.IDLE and age > STALE_TTL_S
                if stuck:
                    logger.warning(f"[FSM] Reaping STUCK conversation {key} (PROCESSING {age:.0f}s).")
                if stuck or stale:
                    self._cancel_debounce_locked(entry)
                    doomed.append(key)
            overflow = len(self._entries) - len(doomed) - MAX_CONVERSATIONS
            if overflow > 0:
                survivors = (k for k in self._entries if k not in doomed)
                oldest = sorted(survivors, key=lambda k: self._entries[k].last_activity)
                doomed += oldest[:overflow]
            for key in doomed:
                self._entries.pop(key, None)
            if doomed:
                logger.info(f"[FSM] Reaped {len(doomed)} conversations.")


fsm = ConversationFSM()