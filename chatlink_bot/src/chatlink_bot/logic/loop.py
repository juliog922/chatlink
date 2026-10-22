# chatlink_bot/src/chatlink_bot/logic/loop.py
"""The one loop: every LOOP_SECONDS it answers due conversations, sends what
is queued, and pings the heartbeat.

There is no timer state. A conversation is DUE when it has client messages
newer than both the session cursor (last_answered_id) and the salesman's last
message, and the newest of them is older than RESPONSE_DELAY_MINUTES. That is
derived from the messages table every time, so a restart loses nothing, a
message that arrives while a turn runs is simply still pending, and a
salesman reply cancels coverage just by existing.

Turns run one at a time: the GPU serves one request at a time anyway, and it
removes every lock.
"""
import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from sqlalchemy import and_, select, func

from ..database import AsyncSessionPG
from ..models import Chat, ConversationSession, EmailChat
from .alerts import alert, heartbeat

logger = logging.getLogger("Loop")

LOOP_SECONDS = float(os.getenv("LOOP_SECONDS", "30"))
DELAY = timedelta(minutes=float(os.getenv("RESPONSE_DELAY_MINUTES", "15")))
# Older client messages are left to the salesman instead of answered late.
MAX_PENDING_AGE = timedelta(hours=float(os.getenv("PENDING_MAX_AGE_HOURS", "6")))

Key = Tuple[str, str, str]  # (channel, client_id, user_id)


async def due_conversations(now: datetime) -> List[Key]:
    due: List[Key] = []
    async with AsyncSessionPG() as db:
        for model, channel in ((Chat, "whatsapp"), (EmailChat, "email")):
            client_msg = and_(model.direction == "received", model.is_bot.is_(False))
            salesman_msg = and_(model.direction == "sent", model.is_bot.is_(False))
            rows = (await db.execute(
                select(model.chat_id, model.user,
                       func.max(model.id).filter(client_msg),
                       func.max(model.timestamp).filter(client_msg),
                       func.max(model.id).filter(salesman_msg))
                .where(model.timestamp >= now - MAX_PENDING_AGE)
                .group_by(model.chat_id, model.user))).all()
            if not rows:
                continue
            cursors = dict(((s.client_id, s.user_id), s.last_answered_id or 0) for s in (await db.execute(
                select(ConversationSession).where(ConversationSession.channel == channel))).scalars())
            for client_id, user_id, last_client_id, last_client_ts, last_salesman_id in rows:
                if last_client_id is None:
                    continue
                handled = max(cursors.get((client_id, user_id), 0), last_salesman_id or 0)
                if last_client_id > handled and last_client_ts <= now - DELAY:
                    due.append((channel, client_id, user_id))
    return due


async def tick(now: Optional[datetime] = None) -> None:
    from .. import handlers
    now = now or handlers._now_utc()
    for key in await due_conversations(now):
        try:
            await handlers.run_turn(*key)
        except Exception as e:
            logger.exception(f"Turn failed for {key}: {e}")
            alert(f"turn:{key[0]}:{key[2]}", f"A turn failed for client {key[1]} of {key[2]} ({key[0]}): {e!r}. "
                  "It is retried every loop; the salesman should answer meanwhile.")
    await handlers.send_queued()
    await handlers.send_orders()


async def run_forever(stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await tick()
            heartbeat(True)
        except Exception as e:
            logger.exception(f"Loop iteration failed: {e}")
            heartbeat(False)
            alert("loop", f"The main loop failed: {e!r}")
        try:
            await asyncio.wait_for(stop.wait(), timeout=LOOP_SECONDS)
        except asyncio.TimeoutError:
            pass
