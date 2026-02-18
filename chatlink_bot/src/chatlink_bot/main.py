import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, text

from .database import pg_engine, sql_engine, PGBase, AsyncSessionPG
from .events import event_bus
from .transport.whatsapp import whatsapp_transport
from .transport.email import email_transport
from .handlers import (
    handle_new_message,
    handle_new_email,
    handle_admin_command,
    handle_ai_trigger,
)
from .models import User
from .api.routes import router as api_router
from .api.simulation import router as sim_router

from .ai.qdrant import qdrant_service
from .ai.rag import rag_service

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = "app.log"

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Root Logger
root_logger = logging.getLogger()
root_logger.setLevel(LOG_LEVEL)

# Console Handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)

# File Handler
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
        nxt = nxt + timedelta(days=1)
    return (nxt - now).total_seconds()


async def _startup_ingest_products() -> None:
    for attempt in range(3):
        try:
            total = await qdrant_service.ingest_products_from_sqlserver()
            root_logger.info(f"Startup ingestion done. Upserted: {total}")
            await rag_service.initialize()
            break  # Success! Exit the loop.
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
                f"Next daily ingestion scheduled in {int(sleep_s)}s "
                f"(UTC {DAILY_INGEST_HOUR_UTC:02d}:{DAILY_INGEST_MINUTE_UTC:02d})"
            )
            try:
                await asyncio.wait_for(stop_evt.wait(), timeout=sleep_s)
                break  # stop requested
            except asyncio.TimeoutError:
                pass  # time to run

            total = await qdrant_service.ingest_products_from_sqlserver()
            root_logger.info(f"Daily ingestion done. Upserted: {total}")
            await rag_service.initialize()
        except Exception as e:
            root_logger.error(f"Daily ingestion loop error: {e}")
            # backoff a bit
            try:
                await asyncio.wait_for(stop_evt.wait(), timeout=60)
                break
            except asyncio.TimeoutError:
                continue

async def _ensure_simulation_actors():
    """Ensures a default Salesman and Admin exist for simulation purposes."""
    async with AsyncSessionPG() as db:
        # 1. Sim Salesman
        sales = (await db.execute(select(User).where(User.email == "sales@sim.local"))).scalars().first()
        if not sales:
            db.add(User(name="Sim Salesman", email="sales@sim.local", phone="34600999001", role="user", enabled=True))
            root_logger.info("Created Simulation Salesman (sales@sim.local)")
        
        # 2. Sim Admin
        admin = (await db.execute(select(User).where(User.email == "admin@sim.local"))).scalars().first()
        if not admin:
            db.add(User(name="Sim Admin", email="admin@sim.local", phone="34600999002", role="admin", enabled=True))
            root_logger.info("Created Simulation Admin (admin@sim.local)")
        
        await db.commit()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1) Create Postgres tables
    async with pg_engine.begin() as conn:
        await conn.run_sync(PGBase.metadata.create_all)

    await _ensure_simulation_actors()

    # 2) Subscribe handlers
    await event_bus.subscribe("message_received", handle_new_message)
    await event_bus.subscribe("email_received", handle_new_email)
    await event_bus.subscribe("admin_command", handle_admin_command)
    await event_bus.subscribe("trigger_ai_processing", handle_ai_trigger)

    # 3) Start transports
    whatsapp_transport.start()
    email_transport.start()

    # 4) Optional DB sanity checks
    try:
        async with sql_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        root_logger.info("SQL Server connectivity: OK")
    except Exception as e:
        root_logger.warning(f"SQL Server connectivity: FAILED ({e})")

    # 5) Start background ingestion tasks
    app.state.stop_daily_ingest = asyncio.Event()
    app.state.startup_ingest_task = asyncio.create_task(_startup_ingest_products())
    app.state.daily_ingest_task = asyncio.create_task(_daily_ingest_loop(app.state.stop_daily_ingest))

    yield

    # Shutdown
    whatsapp_transport.stop()
    email_transport.stop()

    try:
        app.state.stop_daily_ingest.set()
        for t in [getattr(app.state, "startup_ingest_task", None), getattr(app.state, "daily_ingest_task", None)]:
            if t and not t.done():
                t.cancel()
    except Exception:
        pass


app = FastAPI(title="ChatLink Unified Bot", version="3.0.0", lifespan=lifespan)

static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Mount API
app.include_router(api_router, prefix="/api")
app.include_router(sim_router, prefix="/api/test")

from fastapi.responses import FileResponse
@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(static_dir, "index.html"))

if __name__ == "__main__":
    import uvicorn

    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("chatlink_bot.main:app", host=API_HOST, port=API_PORT, reload=False)
