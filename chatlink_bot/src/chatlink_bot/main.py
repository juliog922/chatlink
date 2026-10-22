import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from sqlalchemy import select, text

from .ai.cima_client import CimaClient
from .api.routes import router as api_router
from sqlalchemy import text
from .database import AsyncSessionPG, PGBase, pg_engine, sql_engine
from .events import event_bus
from .handlers import (
    handle_admin_command,
    handle_ai_trigger,
    handle_new_email,
    handle_new_message,
)
from .logic.fsm import fsm
from .models import User
from .transport.email import email_transport
from .transport.whatsapp import whatsapp_transport

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = "app.log"

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
root_logger = logging.getLogger()
root_logger.setLevel(LOG_LEVEL)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

async def _ensure_models_ready() -> None:
    """
    Wait for cima to finish its startup model pull (compose sets
    CIMA_PULL_AT_STARTUP; /api/ready returns 503 until the model is on disk).
    """
    cima_url = os.getenv("CIMA_URL", "http://cima:8000")
    model = os.getenv("CIMA_MODEL", "unsloth/gemma-4-E2B-it-GGUF:Q8_0")
    client = CimaClient(base_url=cima_url, timeout=10.0)
    root_logger.info(f"Waiting for cima to be ready with model: {model}")

    max_wait_s = int(os.getenv("CIMA_READY_MAX_WAIT_S", "1800"))
    waited, poll_s = 0, 10
    while waited < max_wait_s:
        try:
            info = await asyncio.to_thread(client.ready, [model])
            if info.get("ready"):
                root_logger.info("SUCCESS: cima is ready and the model is loaded.")
                return
            root_logger.info(f"cima not ready yet (model still loading). Waited {waited}s.")
        except Exception as e:
            root_logger.warning(f"[cima] Not reachable yet (booting?). Retrying in {poll_s}s... ({e})")
        await asyncio.sleep(poll_s)
        waited += poll_s

    root_logger.error(f"cima not ready within {max_wait_s}s; continuing (AI calls may fail).")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1) Postgres tables
    async with pg_engine.begin() as conn:
        await conn.run_sync(PGBase.metadata.create_all)
        # create_all never ALTERs an existing table, so columns added to a
        # model after its table was first created are missing on upgrade.
        # These IF NOT EXISTS statements are idempotent and cheap; they bring
        # conversation_sessions up to the current model without Alembic.
        await conn.execute(text(
            "ALTER TABLE conversation_sessions "
            "ADD COLUMN IF NOT EXISTS open_items JSONB DEFAULT '[]'::jsonb"))
        await conn.execute(text(
            "ALTER TABLE conversation_sessions "
            "ADD COLUMN IF NOT EXISTS guide_shown BOOLEAN DEFAULT FALSE"))

    # 2) Event handlers
    await event_bus.subscribe("message_received", handle_new_message)
    await event_bus.subscribe("email_received", handle_new_email)
    await event_bus.subscribe("admin_command", handle_admin_command)
    await event_bus.subscribe("trigger_ai_processing", handle_ai_trigger)

    # 3) Transports + FSM janitor
    whatsapp_transport.start()
    email_transport.start()
    fsm.start_cleanup_loop()

    # 4) DB sanity check
    try:
        async with sql_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        root_logger.info("SQL Server connectivity: OK")
    except Exception as e:
        root_logger.warning(f"SQL Server connectivity: FAILED ({e})")

    await _ensure_models_ready()


    yield

    # Shutdown
    whatsapp_transport.stop()
    email_transport.stop()
    fsm.stop_cleanup_loop()


app = FastAPI(title="ChatLink Unified Bot", version="3.1.0", lifespan=lifespan)

app.include_router(api_router, prefix="/api")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("chatlink_bot.main:app",
                host=os.getenv("API_HOST", "0.0.0.0"),
                port=int(os.getenv("API_PORT", "8000")),
                reload=False)