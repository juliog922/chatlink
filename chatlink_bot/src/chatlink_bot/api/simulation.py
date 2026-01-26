import asyncio
import logging
import time
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import select, func

from ..database import AsyncSessionPG, AsyncSessionSQL
from ..models import User, MSClient
from ..handlers import handle_new_message, handle_new_email, handle_ai_trigger
from ..ai.llm import get_summary_state
from ..transport.whatsapp import WhatsAppMessage

router = APIRouter()
logger = logging.getLogger("Simulation")

class SimActor(BaseModel):
    id: str
    name: str
    type: Literal["admin", "user", "client", "non_client"]
    channel_pref: str = "whatsapp" # whatsapp or email

class ActorListResponse(BaseModel):
    actors: List[SimActor]

class SimMessageRequest(BaseModel):
    channel: Literal["whatsapp", "email"]
    sender: str         
    receiver: str       
    text: str
    media_type: str = "text"
    force_ai: bool = False
    # If true, handlers will FAKE the client if not found. 
    # If false, handlers will DROP the message if not found (testing "Not Client" case).
    mock_client_force: bool = False 

class SimStateResponse(BaseModel):
    order_status: str
    confirmed_items: List[Dict[str, Any]]
    chat_context_summary: str
    last_benchmark_ms: float = 0.0

@router.get("/actors", response_model=ActorListResponse, tags=["Simulation"])
async def get_simulation_actors():
    actors = []

    # 1. Internal Users (Admin & Salesman) from Postgres
    async with AsyncSessionPG() as db:
        users = (await db.execute(select(User).where(User.enabled == True))).scalars().all()
        for u in users:
            actors.append(SimActor(
                id=u.phone if u.phone and len(u.phone)>5 else u.email,
                name=f"{u.name} ({u.role})",
                type="admin" if u.role == "admin" else "user",
                channel_pref="email" if "@" in u.email and not u.phone else "whatsapp"
            ))

    # 2. Real Clients from SQL Server (Sample 5)
    try:
        async with AsyncSessionSQL() as db:
            stmt = select(MSClient).limit(5)
            clients = (await db.execute(stmt)).scalars().all()
            for c in clients:
                phone = (c.Telefono or c.Telefono2 or "").strip().replace(" ", "")
                email = (c.EMail1 or "").strip()
                
                # Add as WA actor
                if phone:
                    actors.append(SimActor(id=phone, name=f"[Client] {c.Nombre}", type="client", channel_pref="whatsapp"))
                # Add as Email actor
                if email:
                    actors.append(SimActor(id=email, name=f"[Client] {c.Nombre}", type="client", channel_pref="email"))
    except Exception as e:
        logger.warning(f"Could not fetch clients for sim: {e}")

    # 3. Common Not-Client (Static/Random)
    actors.append(SimActor(id="34600000404", name="Random Stranger (Phone)", type="non_client", channel_pref="whatsapp"))
    actors.append(SimActor(id="stranger@unknown.com", name="Random Stranger (Email)", type="non_client", channel_pref="email"))

    return {"actors": actors}

@router.post("/message", tags=["Simulation"])
async def simulate_message(req: SimMessageRequest):
    start_time = time.perf_counter()
    logger.info(f"[SIM] {req.channel}: {req.sender} -> {req.receiver} (MockClient={req.mock_client_force})")

    # MOCK MEDIA 
    binary_data = b""
    filename = ""
    if req.media_type != "text":
        if req.media_type == "image":
            binary_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
            filename = "sim_image.png"
        elif req.media_type == "audio":
            binary_data = b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
            filename = "sim_audio.wav"
        else:
            binary_data = b'%PDF-1.4\n%...\n%%EOF'
            filename = "sim_doc.pdf"

    if req.channel == "whatsapp":
        mock_msg = WhatsAppMessage(
            raw=None,
            from_jid=f"{req.sender}@s.whatsapp.net",
            to_jid=f"{req.receiver}@s.whatsapp.net",
            from_phone=req.sender,
            to_phone=req.receiver,
            name="Sim User",
            text=req.text,
            timestamp=str(datetime.now().timestamp()),
            binary=binary_data,
            filename=filename
        )
        await handle_new_message({
            "normalized": mock_msg, 
            "is_simulation": True, 
            "mock_client_force": req.mock_client_force
        })

    elif req.channel == "email":
        attachments = []
        if binary_data:
            attachments.append({"filename": filename, "content_type": "application/octet-stream", "bytes": binary_data})

        await handle_new_email({
            "user_mailbox": req.receiver, 
            "from": req.sender,
            "to": req.receiver,
            "subject": "Sim Msg",
            "body": req.text,
            "attachments": attachments,
            "direction": "received",
            "is_simulation": True,
            "mock_client_force": req.mock_client_force
        })

    if req.force_ai:
        logger.info("[SIM] 'force_ai' flag present but ignored to respect FSM debounce logic.")

    end_time = time.perf_counter()
    return {"status": "ok", "benchmark_ms": round((end_time - start_time) * 1000, 2)}

@router.get("/state/{channel}/{client_id}", response_model=SimStateResponse, tags=["Simulation"])
async def get_simulation_state(channel: str, client_id: str):
    state = get_summary_state(client_id)
    summary = state.get("summary", {})
    return SimStateResponse(
        order_status=summary.get("order_status", "IDLE"),
        confirmed_items=summary.get("confirmed_items", []) or [],
        chat_context_summary=summary.get("chat_context_summary", "")
    )