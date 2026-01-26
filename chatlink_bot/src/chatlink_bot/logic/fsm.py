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


class ConversationState(str, Enum):
    IDLE = "IDLE"
    DEBOUNCING = "DEBOUNCING"
    PROCESSING = "PROCESSING"


@dataclass
class ConversationEntry:
    state: ConversationState = ConversationState.IDLE
    last_client_msg_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    debounce_token: int = 0
    debounce_task: Optional[asyncio.Task] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    is_simulation: bool = False  # [Modified] Added to track simulation status


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def conversation_key(channel: str, client_id: str, user_id: str) -> Tuple[str, str, str]:
    return (channel, client_id, user_id)


class ConversationFSM:
    def __init__(self, debounce_minutes: int = DEBOUNCE_MINUTES) -> None:
        self.debounce_seconds = int(debounce_minutes * 60)
        self._conversations: Dict[Tuple[str, str, str], ConversationEntry] = {}
        self._map_lock = asyncio.Lock()

    async def _get_entry(self, key: Tuple[str, str, str]) -> ConversationEntry:
        async with self._map_lock:
            if key not in self._conversations:
                self._conversations[key] = ConversationEntry()
            return self._conversations[key]

    async def on_client_message(self, channel: str, client_id: str, user_id: str, is_simulation: bool = False) -> None:
        key = conversation_key(channel, client_id, user_id)
        entry = await self._get_entry(key)

        async with entry.lock:
            entry.last_client_msg_at = _now_utc()
            entry.state = ConversationState.DEBOUNCING
            entry.is_simulation = is_simulation  # [Modified] Store flag
            await self._cancel_debounce_locked(entry)
            await self._start_debounce_locked(key, entry)

    async def on_user_message(self, channel: str, client_id: str, user_id: str) -> None:
        key = conversation_key(channel, client_id, user_id)
        entry = await self._get_entry(key)

        async with entry.lock:
            await self._cancel_debounce_locked(entry)
            entry.state = ConversationState.IDLE

    async def on_ai_done(self, channel: str, client_id: str, user_id: str) -> None:
        key = conversation_key(channel, client_id, user_id)
        entry = await self._get_entry(key)

        async with entry.lock:
            if entry.state == ConversationState.PROCESSING:
                entry.state = ConversationState.IDLE

    async def _cancel_debounce_locked(self, entry: ConversationEntry) -> None:
        if entry.debounce_task and not entry.debounce_task.done():
            entry.debounce_task.cancel()
        entry.debounce_task = None

    async def _start_debounce_locked(self, key: Tuple[str, str, str], entry: ConversationEntry) -> None:
        entry.debounce_token += 1
        token = entry.debounce_token
        entry.debounce_task = asyncio.create_task(self._debounce_timer(key, token))

        ch, client, user = key
        # logger.info(f"[{ch}] DEBOUNCING started/reset for client={client} user={user} token={token}")

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
            sim_flag = entry.is_simulation  # [Modified] Retrieve flag safely

        ch, client, user = key
        logger.info(f"[{ch}] Timer fired -> PROCESSING client={client} user={user} token={token} (sim={sim_flag})")

        await event_bus.emit(
            "trigger_ai_processing",
            {
                "channel": ch,
                "client_id": client,
                "user_id": user,
                "fired_at": _now_utc().isoformat(),
                "is_simulation": sim_flag, # [Modified] Pass flag to AI handler
            },
        )


fsm = ConversationFSM()