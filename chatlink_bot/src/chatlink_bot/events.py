# chatlink_bot/src/chatlink_bot/events.py
import asyncio
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("EventBus")


@dataclass(frozen=True)
class Subscriber:
    handler: Callable
    run_sync_in_thread: bool = True
    background: bool = False  # if True: fire-and-forget (never awaited)


class EventBus:
    """
    Simple Pub/Sub spine.

    Improvements vs current:
    - runs async handlers concurrently
    - optionally runs sync handlers in a thread (avoids blocking the loop)
    - unsubscribe support
    - background (fire-and-forget) subscriptions for non-critical side-effects
    """
    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Subscriber]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(
        self,
        event_name: str,
        handler: Callable,
        *,
        run_sync_in_thread: bool = True,
        background: bool = False,
    ) -> None:
        async with self._lock:
            self._subscribers.setdefault(event_name, []).append(
                Subscriber(handler=handler, run_sync_in_thread=run_sync_in_thread, background=background)
            )
        logger.info(f"Subscribed {getattr(handler, '__name__', str(handler))} to {event_name}")

    async def unsubscribe(self, event_name: str, handler: Callable) -> None:
        async with self._lock:
            if event_name not in self._subscribers:
                return
            self._subscribers[event_name] = [s for s in self._subscribers[event_name] if s.handler != handler]
            if not self._subscribers[event_name]:
                self._subscribers.pop(event_name, None)

    async def emit(self, event_name: str, payload: Any) -> None:
        async with self._lock:
            subs = list(self._subscribers.get(event_name, []))

        if not subs:
            return

        tasks: List[asyncio.Task] = []
        awaitables: List[Any] = []

        for sub in subs:
            h = sub.handler
            try:
                if inspect.iscoroutinefunction(h):
                    coro = h(payload)
                    if sub.background:
                        tasks.append(asyncio.create_task(self._safe_coro(event_name, h, coro)))
                    else:
                        awaitables.append(self._safe_coro(event_name, h, coro))
                else:
                    if sub.run_sync_in_thread:
                        coro = asyncio.to_thread(h, payload)
                        if sub.background:
                            tasks.append(asyncio.create_task(self._safe_coro(event_name, h, coro)))
                        else:
                            awaitables.append(self._safe_coro(event_name, h, coro))
                    else:
                        # Runs inline (can block); only for tiny handlers
                        try:
                            h(payload)
                        except Exception as e:
                            logger.error(f"Error in sync handler {h} for {event_name}: {e}")
            except Exception as e:
                logger.error(f"Error scheduling handler {h} for {event_name}: {e}")

        if awaitables:
            # run concurrently, keep going even if one fails
            await asyncio.gather(*awaitables, return_exceptions=True)

        # background tasks are already scheduled; no await

    async def _safe_coro(self, event_name: str, handler: Callable, coro: Any) -> None:
        try:
            await coro
        except Exception as e:
            logger.error(f"Error in handler {getattr(handler, '__name__', str(handler))} for {event_name}: {e}")


event_bus = EventBus()
