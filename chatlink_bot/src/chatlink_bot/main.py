# chatlink_bot/src/chatlink_bot/main.py
import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.routes import router as api_router
from .database import init_schema
from .events import event_bus
from .handlers import handle_admin_command, handle_new_email, handle_new_message, resume_mailboxes
from .logic.loop import run_forever
from .transport.email import email_transport
from .transport.whatsapp import whatsapp_transport

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_schema()
    await event_bus.subscribe("message_received", handle_new_message)
    await event_bus.subscribe("email_received", handle_new_email)
    await event_bus.subscribe("admin_command", handle_admin_command)
    whatsapp_transport.start()
    email_transport.start()
    await resume_mailboxes()
    stop = asyncio.Event()
    loop_task = asyncio.create_task(run_forever(stop))
    yield
    stop.set()
    await loop_task
    whatsapp_transport.stop()
    email_transport.stop()


app = FastAPI(title="Kapa", version="3.0.0", lifespan=lifespan)
app.include_router(api_router, prefix="/api")
