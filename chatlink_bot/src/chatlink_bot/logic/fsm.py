import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional, Tuple

from ..events import event_bus

logger = logging.getLogger("FSM")

DEBOUNCE_MINUTES = int(os.getenv("RESPONSE_DELAY_MINUTES", "15"))

# ---------------------------------------------------------------------------
# Safety valves (env-configurable)
# ---------------------------------------------------------------------------
# Max seconds AI processing is allowed before we force-reset to IDLE.
# Protects against handler crashes / deadlocks leaving a conversation stuck.
PROCESSING_TIMEOUT_S = int(os.getenv("FSM_PROCESSING_TIMEOUT_S", "300"))

# Conversations sitting IDLE with no activity for this many seconds get reaped.
STALE_TTL_S = int(os.getenv("FSM_STALE_TTL_S", "3600"))

# How often the background janitor runs (seconds).
CLEANUP_INTERVAL_S = int(os.getenv("FSM_CLEANUP_INTERVAL_S", "120"))

# Max tracked conversations before forced eviction of oldest entries.
MAX_CONVERSATIONS = int(os.getenv("FSM_MAX_CONVERSATIONS", "5000"))


class ConversationState(str, Enum):
    IDLE = "IDLE"
    DEBOUNCING = "DEBOUNCING"
    PROCESSING = "PROCESSING"


@dataclass
class ConversationEntry:
    state: ConversationState = ConversationState.IDLE
    last_client_msg_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    debounce_token: int = 0
    debounce_task: Optional[asyncio.Task] = None
    processing_watchdog: Optional[asyncio.Task] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    is_simulation: bool = False

    # If the client sends more messages while the AI is still working,
    # we record it so we can re-debounce after processing finishes.
    pending_redebounce: bool = False


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def conversation_key(channel: str, client_id: str, user_id: str) -> Tuple[str, str, str]:
    return (channel, client_id, user_id)


class ConversationFSM:
    def __init__(self, debounce_minutes: int = DEBOUNCE_MINUTES) -> None:
        self.debounce_seconds = int(debounce_minutes * 60)
        self._conversations: Dict[Tuple[str, str, str], ConversationEntry] = {}
        self._map_lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Background janitor
    # ------------------------------------------------------------------

    def start_cleanup_loop(self) -> None:
        """Call once after the event loop is running (e.g. in lifespan)."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    def stop_cleanup_loop(self) -> None:
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()

    async def _cleanup_loop(self) -> None:
        """Periodically reap stale and stuck conversations."""
        while True:
            try:
                await asyncio.sleep(CLEANUP_INTERVAL_S)
                await self._reap_stale()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"FSM cleanup loop error: {e}")

    async def _reap_stale(self) -> None:
        now = _now_utc()
        to_remove: list[Tuple[str, str, str]] = []

        async with self._map_lock:
            for key, entry in list(self._conversations.items()):
                age = (now - entry.last_activity_at).total_seconds()

                # Stuck in PROCESSING beyond the timeout
                if entry.state == ConversationState.PROCESSING and age > PROCESSING_TIMEOUT_S:
                    logger.warning(
                        f"[FSM] Force-resetting STUCK conversation {key} "
                        f"(PROCESSING for {age:.0f}s > {PROCESSING_TIMEOUT_S}s)"
                    )
                    await self._cancel_tasks_locked(entry)
                    entry.state = ConversationState.IDLE
                    to_remove.append(key)

                # IDLE with no recent activity
                elif entry.state == ConversationState.IDLE and age > STALE_TTL_S:
                    to_remove.append(key)

            for key in to_remove:
                self._conversations.pop(key, None)

            # Hard cap: evict oldest if we're over the limit
            if len(self._conversations) > MAX_CONVERSATIONS:
                overflow = len(self._conversations) - MAX_CONVERSATIONS
                sorted_keys = sorted(
                    self._conversations.keys(),
                    key=lambda k: self._conversations[k].last_activity_at,
                )
                for key in sorted_keys[:overflow]:
                    entry = self._conversations.pop(key, None)
                    if entry:
                        await self._cancel_tasks_locked(entry)

        if to_remove:
            logger.info(f"[FSM] Reaped {len(to_remove)} stale conversations.")

    # ------------------------------------------------------------------
    # Entry management
    # ------------------------------------------------------------------

    async def _get_entry(self, key: Tuple[str, str, str]) -> ConversationEntry:
        async with self._map_lock:
            if key not in self._conversations:
                self._conversations[key] = ConversationEntry()
            return self._conversations[key]

    async def _remove_entry(self, key: Tuple[str, str, str]) -> None:
        async with self._map_lock:
            self._conversations.pop(key, None)

    # ------------------------------------------------------------------
    # Public interface (unchanged signatures)
    # ------------------------------------------------------------------

    async def on_client_message(
        self, channel: str, client_id: str, user_id: str, is_simulation: bool = False
    ) -> None:
        key = conversation_key(channel, client_id, user_id)
        entry = await self._get_entry(key)

        async with entry.lock:
            now = _now_utc()
            entry.last_client_msg_at = now
            entry.last_activity_at = now
            entry.is_simulation = is_simulation

            if entry.state == ConversationState.PROCESSING:
                # AI is working right now.  Don't cancel it — just flag that
                # a new message arrived so we re-debounce after it finishes.
                entry.pending_redebounce = True
                logger.info(
                    f"[FSM] Client msg during PROCESSING for {key}; will re-debounce after AI finishes."
                )
                return

            # Normal path: cancel any existing debounce and start fresh
            entry.state = ConversationState.DEBOUNCING
            await self._cancel_debounce_locked(entry)
            await self._start_debounce_locked(key, entry)

    async def on_user_message(self, channel: str, client_id: str, user_id: str) -> None:
        """Salesman replied manually → cancel everything, go IDLE, remove entry."""
        key = conversation_key(channel, client_id, user_id)
        entry = await self._get_entry(key)

        async with entry.lock:
            entry.last_activity_at = _now_utc()
            await self._cancel_tasks_locked(entry)
            entry.state = ConversationState.IDLE
            entry.pending_redebounce = False

        await self._remove_entry(key)

    async def on_ai_done(self, channel: str, client_id: str, user_id: str) -> None:
        """AI handler finished.  Either go IDLE or re-debounce if client sent more."""
        key = conversation_key(channel, client_id, user_id)
        entry = await self._get_entry(key)

        async with entry.lock:
            entry.last_activity_at = _now_utc()

            # Cancel the processing watchdog
            if entry.processing_watchdog and not entry.processing_watchdog.done():
                entry.processing_watchdog.cancel()
            entry.processing_watchdog = None

            if entry.state != ConversationState.PROCESSING:
                # Already cancelled by salesman or cleanup — nothing to do
                return

            if entry.pending_redebounce:
                # Client sent more messages while AI was working → restart debounce
                entry.pending_redebounce = False
                entry.state = ConversationState.DEBOUNCING
                await self._cancel_debounce_locked(entry)
                await self._start_debounce_locked(key, entry)
                logger.info(f"[FSM] Re-debouncing after AI done for {key} (client sent more messages).")
                return

            # Normal completion → IDLE
            entry.state = ConversationState.IDLE

        # Clean up if truly idle
        if entry.state == ConversationState.IDLE:
            await self._remove_entry(key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _cancel_debounce_locked(self, entry: ConversationEntry) -> None:
        """Cancel the debounce timer task.  Caller must hold entry.lock."""
        if entry.debounce_task and not entry.debounce_task.done():
            entry.debounce_task.cancel()
        entry.debounce_task = None

    async def _cancel_tasks_locked(self, entry: ConversationEntry) -> None:
        """Cancel ALL background tasks on an entry.  Caller must hold entry.lock."""
        await self._cancel_debounce_locked(entry)
        if entry.processing_watchdog and not entry.processing_watchdog.done():
            entry.processing_watchdog.cancel()
        entry.processing_watchdog = None

    async def _start_debounce_locked(
        self, key: Tuple[str, str, str], entry: ConversationEntry
    ) -> None:
        """Start a new debounce timer.  Caller must hold entry.lock."""
        entry.debounce_token += 1
        token = entry.debounce_token
        entry.debounce_task = asyncio.create_task(self._debounce_timer(key, token))

    async def _debounce_timer(self, key: Tuple[str, str, str], token: int) -> None:
        try:
            await asyncio.sleep(self.debounce_seconds)
        except asyncio.CancelledError:
            return

        entry = await self._get_entry(key)

        sim_flag = False
        async with entry.lock:
            if entry.debounce_token != token:
                return
            if entry.state != ConversationState.DEBOUNCING:
                return

            entry.state = ConversationState.PROCESSING
            entry.last_activity_at = _now_utc()
            entry.pending_redebounce = False
            sim_flag = entry.is_simulation

            # Start a watchdog that will force-reset to IDLE if the AI
            # handler never calls on_ai_done (crash / timeout protection).
            entry.processing_watchdog = asyncio.create_task(
                self._processing_watchdog(key, token)
            )

        ch, client, user = key
        logger.info(
            f"[{ch}] Timer fired -> PROCESSING client={client} user={user} "
            f"token={token} (sim={sim_flag})"
        )

        await event_bus.emit(
            "trigger_ai_processing",
            {
                "channel": ch,
                "client_id": client,
                "user_id": user,
                "fired_at": _now_utc().isoformat(),
                "is_simulation": sim_flag,
            },
        )

    async def _processing_watchdog(self, key: Tuple[str, str, str], token: int) -> None:
        """Safety net: force IDLE if AI never finishes within the timeout."""
        try:
            await asyncio.sleep(PROCESSING_TIMEOUT_S)
        except asyncio.CancelledError:
            return  # Normal path: on_ai_done cancelled us

        entry = await self._get_entry(key)
        async with entry.lock:
            if entry.debounce_token != token:
                return
            if entry.state != ConversationState.PROCESSING:
                return

            logger.error(
                f"[FSM] WATCHDOG: AI processing timed out for {key} after "
                f"{PROCESSING_TIMEOUT_S}s. Force-resetting to IDLE."
            )
            entry.state = ConversationState.IDLE
            entry.pending_redebounce = False

        await self._remove_entry(key)



fsm = ConversationFSM()