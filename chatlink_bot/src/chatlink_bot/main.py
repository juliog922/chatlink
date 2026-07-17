import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, text

from .ai.cima_client import CimaClient
from .ai.qdrant import qdrant_service
from .ai.rag import rag_service
from .api.routes import router as api_router
from .api.simulation import router as sim_router
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

DAILY_INGEST_HOUR_UTC = int(os.getenv("QDRANT_DAILY_INGEST_HOUR_UTC", "3"))
DAILY_INGEST_MINUTE_UTC = int(os.getenv("QDRANT_DAILY_INGEST_MINUTE_UTC", "0"))


def _seconds_until_next_daily(hour: int, minute: int) -> float:
    now = datetime.now(timezone.utc)
    nxt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if nxt <= now:
        nxt += timedelta(days=1)
    return (nxt - now).total_seconds()


async def _sync_catalog(reason: str) -> None:
    """
    Incremental SAGE -> Qdrant sync (hash-diff: only new/changed products are
    embedded, stale ones deleted — see ai/qdrant.py). Cheap when nothing
    changed, so it runs at EVERY startup: the old count>0 skip meant products
    added to SAGE were invisible until the next 3AM job. The BM25 index is
    rebuilt only when the catalog actually changed (or on first load).
    """
    written = await qdrant_service.ingest_products_from_sqlserver()
    root_logger.info(f"[Ingest:{reason}] Sync finished: {written} points written.")
    if written > 0 or rag_service.bm25 is None:
        await rag_service.initialize()


async def _startup_ingest_products() -> None:
    for attempt in range(3):
        try:
            await qdrant_service.ensure_ready()
            await _sync_catalog("startup")
            return
        except Exception as e:
            if attempt < 2:
                root_logger.warning(f"Startup ingestion failed, retrying in 5s... ({e})")
                await asyncio.sleep(5)
            else:
                root_logger.error(f"Startup ingestion completely failed: {e}")


async def _daily_ingest_loop(stop_evt: asyncio.Event) -> None:
    while not stop_evt.is_set():
        try:
            sleep_s = _seconds_until_next_daily(DAILY_INGEST_HOUR_UTC, DAILY_INGEST_MINUTE_UTC)
            root_logger.info(
                f"Next daily ingestion in {int(sleep_s)}s "
                f"(UTC {DAILY_INGEST_HOUR_UTC:02d}:{DAILY_INGEST_MINUTE_UTC:02d})")
            try:
                await asyncio.wait_for(stop_evt.wait(), timeout=sleep_s)
                return  # stop requested
            except asyncio.TimeoutError:
                pass  # time to run
            await _sync_catalog("daily")
        except Exception as e:
            root_logger.error(f"Daily ingestion loop error: {e}")
            try:
                await asyncio.wait_for(stop_evt.wait(), timeout=60)
                return
            except asyncio.TimeoutError:
                continue


async def _ensure_simulation_actors() -> None:
    """Ensure the default sim Salesman and Admin exist."""
    async with AsyncSessionPG() as db:
        sales = (await db.execute(select(User).where(User.email == "sales@sim.com"))).scalars().first()
        if not sales:
            db.add(User(name="Sim Salesman", email="sales@sim.com", phone="34600999001",
                        role="user", enabled=True))
        admin = (await db.execute(select(User).where(User.email == "admin@sim.com"))).scalars().first()
        if not admin:
            db.add(User(name="Sim Admin", email="admin@sim.com", phone="34600999002",
                        role="admin", enabled=True))
        await db.commit()


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
    await _ensure_simulation_actors()

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

    # 5) Catalog sync: incremental at startup + daily
    app.state.stop_daily_ingest = asyncio.Event()
    app.state.startup_ingest_task = asyncio.create_task(_startup_ingest_products())
    app.state.daily_ingest_task = asyncio.create_task(_daily_ingest_loop(app.state.stop_daily_ingest))

    yield

    # Shutdown
    whatsapp_transport.stop()
    email_transport.stop()
    fsm.stop_cleanup_loop()
    try:
        app.state.stop_daily_ingest.set()
        for t in (getattr(app.state, "startup_ingest_task", None),
                  getattr(app.state, "daily_ingest_task", None)):
            if t and not t.done():
                t.cancel()
    except Exception:
        pass


app = FastAPI(title="ChatLink Unified Bot", version="3.1.0", lifespan=lifespan)

static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

app.include_router(api_router, prefix="/api")
app.include_router(sim_router, prefix="/api/test")


@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(static_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("chatlink_bot.main:app",
                host=os.getenv("API_HOST", "0.0.0.0"),
                port=int(os.getenv("API_PORT", "8000")),
                reload=False)