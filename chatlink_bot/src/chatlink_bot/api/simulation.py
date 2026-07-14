# chatlink_bot/src/chatlink_bot/api/simulation.py
"""
Development simulator.

The old version injected messages and left you polling Postgres while a real
15-minute debounce ran — testing required linked phones and patience. Now:

  POST /api/test/message   inject only (kept for the existing dashboard UI)
  POST /api/test/flush     fire a pending debounce NOW (skip the wait)
  POST /api/test/turn      inject + flush + return the bot's actual reply,
                           synchronously, in one call — the whole pipeline
                           (gatekeeper, FSM, agent, RAG) runs for real; only
                           time and the transport delivery are bypassed.
  GET  /api/test/state     persisted ConversationSession (single source of truth)

MODES (unchanged semantics)
  logic    fake salesman ↔ fake client; nothing real needed. Replies are
           captured (and the real send fails harmlessly: no linked device).
  channel  sim salesman ↔ your REAL phone/mailbox; receiver must be a linked
           enabled user so the reply leaves through the real transport.
  full     one linked account messaging itself (self-chat logic in handlers);
           nothing to call here — /modes just reports readiness.
"""
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from ..database import AsyncSessionPG, AsyncSessionSQL
from ..handlers import handle_new_email, handle_new_message
from ..logic.fsm import fsm
from ..logic.sim_clients import sim_client_cache
from ..models import ConversationSession, MSClient, User
from ..transport.outbox import sim_capture
from ..transport.whatsapp import WhatsAppMessage, whatsapp_transport

router = APIRouter()
logger = logging.getLogger("Simulation")

SIM_SALES_EMAIL = "sales@sim.com"
SIM_SALES_PHONE = "34600999001"

# Tiny valid media samples (decodable 1x1 PNG, minimal PCM WAV, minimal PDF).
_SAMPLE_PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
               b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
               b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")
_SAMPLE_WAV = (b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00"
               b"\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
_SAMPLE_PDF = b"%PDF-1.4\n%...\n%%EOF"


# ---------------------------------------------------------------------- models
class SimActor(BaseModel):
    id: str
    name: str
    type: Literal["admin", "user", "client", "non_client"]
    channel_pref: str = "whatsapp"


class ActorListResponse(BaseModel):
    actors: List[SimActor]


class SimMessageRequest(BaseModel):
    channel: Literal["whatsapp", "email"]
    mode: Literal["logic", "channel"] = "logic"
    sender: str                      # the "client" identity
    receiver: str = ""               # the salesman; defaults to the sim salesman
    text: str
    media_type: str = "text"
    mock_client_force: Optional[bool] = None  # explicit False tests the "not in SAGE -> dropped" path


class SimStateResponse(BaseModel):
    order_status: str
    confirmed_items: List[Dict[str, Any]]
    chat_context_summary: str
    bot_enabled: bool = True
    last_benchmark_ms: float = 0.0
    # Context-window telemetry from the last agent run (est. tokens):
    # window, budget, used, pct_window, pct_budget, history_trimmed, message_truncated
    ctx: Dict[str, Any] = {}


class SimTurnResponse(BaseModel):
    status: str
    fired: bool                      # False = the message never armed the FSM (e.g. gatekeeper drop)
    replies: List[str]               # what the bot answered this turn ("" turns are silence)
    state: SimStateResponse
    benchmark_ms: float


class SimModesResponse(BaseModel):
    modes: Dict[str, str]
    linked_devices: List[str]
    full_mode_ready: bool
    full_mode_hint: str


# --------------------------------------------------------------------- helpers
def _resolve(req: SimMessageRequest) -> tuple[str, bool]:
    receiver = (req.receiver or "").strip() or (SIM_SALES_PHONE if req.channel == "whatsapp" else SIM_SALES_EMAIL)
    mock_force = req.mock_client_force if req.mock_client_force is not None else True
    return receiver, mock_force


def _media(media_type: str) -> tuple[bytes, str]:
    if media_type == "image":
        return _SAMPLE_PNG, "sim_image.png"
    if media_type == "audio":
        return _SAMPLE_WAV, "sim_audio.wav"
    if media_type not in ("", "text"):
        return _SAMPLE_PDF, "sim_doc.pdf"
    return b"", ""


async def _inject(req: SimMessageRequest, receiver: str, mock_force: bool) -> None:
    """Enter the pipeline exactly where real traffic does (the handlers)."""
    binary, filename = _media(req.media_type)
    if req.channel == "whatsapp":
        mock_msg = WhatsAppMessage(
            raw=None, from_jid=f"{req.sender}@s.whatsapp.net", to_jid=f"{receiver}@s.whatsapp.net",
            from_phone=req.sender, to_phone=receiver, name="Sim Client", text=req.text,
            timestamp=str(datetime.now().timestamp()), binary=binary, filename=filename)
        await handle_new_message({"normalized": mock_msg, "is_simulation": True,
                                  "mock_client_force": mock_force})
    else:
        attachments = ([{"filename": filename, "content_type": "application/octet-stream", "bytes": binary}]
                       if binary else [])
        await handle_new_email({"user_mailbox": receiver, "from": req.sender, "to": receiver,
                                "subject": "Sim Msg", "body": req.text, "attachments": attachments,
                                "direction": "received", "is_simulation": True,
                                "mock_client_force": mock_force})


async def _read_state(channel: str, client_id: str, user_id: str = "") -> SimStateResponse:
    async with AsyncSessionPG() as db:
        stmt = select(ConversationSession).where(
            ConversationSession.channel == channel, ConversationSession.client_id == client_id)
        if user_id:
            stmt = stmt.where(ConversationSession.user_id == user_id)
        session = (await db.execute(stmt.order_by(ConversationSession.updated_at.desc()))).scalars().first()
    if session is None:
        return SimStateResponse(order_status="IDLE", confirmed_items=[], chat_context_summary="")
    return SimStateResponse(order_status=session.order_status or "IDLE",
                            confirmed_items=session.cart or [],
                            chat_context_summary=session.summary or "",
                            bot_enabled=bool(session.bot_enabled))


# -------------------------------------------------------------------- endpoints
@router.get("/modes", response_model=SimModesResponse, tags=["Simulation"])
async def get_modes():
    devices = whatsapp_transport.list_devices()
    return SimModesResponse(
        modes={
            "logic": "fake salesman ↔ fake client, no devices needed (tests logic only)",
            "channel": "sim salesman ↔ your real phone/email (tests real channels)",
            "full": "one linked account messaging itself (tests complete workflow)",
        },
        linked_devices=devices,
        full_mode_ready=bool(devices),
        full_mode_hint=("Link a device (login), then WhatsApp your OWN number. Self-chat logic "
                        "answers as the bot; the echo guard prevents loops. Nothing to call here."),
    )


@router.get("/actors", response_model=ActorListResponse, tags=["Simulation"])
async def get_simulation_actors():
    actors: List[SimActor] = []

    async with AsyncSessionPG() as db:  # internal users (sim + real)
        users = (await db.execute(select(User).where(User.enabled == True))).scalars().all()  # noqa: E712
        for u in users:
            actors.append(SimActor(
                id=u.phone if u.phone and len(u.phone) > 5 else u.email,
                name=f"{u.name} ({u.role.value if hasattr(u.role, 'value') else u.role})",
                type="admin" if str(u.role).endswith("admin") or str(u.role).endswith("ADMIN") else "user",
                channel_pref="email" if "@" in (u.email or "") and not u.phone else "whatsapp"))

    try:  # sample of real SAGE clients, for channel-mode realism
        async with AsyncSessionSQL() as db:
            for c in (await db.execute(select(MSClient).limit(5))).scalars().all():
                phone = (c.Telefono or c.Telefono2 or "").strip().replace(" ", "")
                email = (c.EMail1 or "").strip()
                if phone:
                    actors.append(SimActor(id=phone, name=f"[Client] {c.Nombre}", type="client"))
                if email:
                    actors.append(SimActor(id=email, name=f"[Client] {c.Nombre}", type="client", channel_pref="email"))
    except Exception as e:
        logger.warning(f"Could not fetch clients for sim: {e}")

    actors.append(SimActor(id="34600000404", name="Random Stranger (Phone)", type="non_client"))
    actors.append(SimActor(id="stranger@unknown.com", name="Random Stranger (Email)",
                           type="non_client", channel_pref="email"))
    return {"actors": actors}


@router.post("/message", tags=["Simulation"])
async def simulate_message(req: SimMessageRequest):
    """Inject only (the FSM debounce runs at its normal speed). Kept for the dashboard."""
    start = time.perf_counter()
    receiver, mock_force = _resolve(req)
    logger.info(f"[SIM:{req.mode}] {req.channel}: {req.sender} -> {receiver} (mock_client={mock_force})")
    await _inject(req, receiver, mock_force)
    return {"status": "ok", "mode": req.mode,
            "benchmark_ms": round((time.perf_counter() - start) * 1000, 2),
            "note": "Reply appears after the FSM debounce fires; use /turn for a synchronous round-trip."}


@router.post("/flush/{channel}/{client_id}/{user_id}", tags=["Simulation"])
async def flush_conversation(channel: str, client_id: str, user_id: str):
    """Fire a pending debounce NOW and await the whole AI pipeline."""
    fired = await fsm.flush(channel, client_id, user_id)
    return {"status": "ok", "fired": fired,
            "replies": [c["text"] for c in sim_capture.drain((channel, client_id, user_id))]}


@router.post("/turn", response_model=SimTurnResponse, tags=["Simulation"])
async def simulate_turn(req: SimMessageRequest):
    """
    One full synchronous turn: inject -> flush -> return the bot's real reply.
    `fired=False` + no replies means the pipeline dropped the message (e.g.
    non-SAGE client with mock_client_force=False) — useful for gatekeeper tests.
    """
    start = time.perf_counter()
    receiver, mock_force = _resolve(req)
    await _inject(req, receiver, mock_force)
    fired = await fsm.flush(req.channel, req.sender, receiver)   # awaits agent + delivery
    captured = sim_capture.drain((req.channel, req.sender, receiver))
    replies = [c["text"] for c in captured]
    state = await _read_state(req.channel, req.sender, receiver)
    state.ctx = next((c["meta"].get("ctx", {}) for c in reversed(captured) if c.get("meta")), {})
    state.last_benchmark_ms = round((time.perf_counter() - start) * 1000, 2)
    return SimTurnResponse(status="ok", fired=fired, replies=replies, state=state,
                           benchmark_ms=state.last_benchmark_ms)


@router.get("/state/{channel}/{client_id}", response_model=SimStateResponse, tags=["Simulation"])
async def get_simulation_state(channel: str, client_id: str):
    """Persisted session state (latest session for this client on this channel)."""
    return await _read_state(channel, client_id)


# --------------------------------------------------------------------------
# Simulation clients: an in-memory CRUD cache checked BEFORE SAGE in every
# client lookup. Test-only — nothing is written to any database, and a
# restart clears the cache. Entries behave downstream exactly like SAGE
# clients (gatekeeper, prompt name, xlsx code).
# --------------------------------------------------------------------------

@router.get("/clients")
async def list_sim_clients():
    return {"clients": sim_client_cache.list()}


@router.post("/clients")
async def upsert_sim_client(payload: dict):
    identifier = str(payload.get("identifier") or "").strip()
    if not identifier:
        raise HTTPException(status_code=422, detail="'identifier' (teléfono o email) es obligatorio")
    entry = sim_client_cache.upsert(
        identifier,
        name=(payload.get("name") or "").strip() or None,
        code=(payload.get("code") or "").strip() or None,
        notes=str(payload.get("notes") or ""),
    )
    return {"ok": True, "client": entry}


@router.delete("/clients/{identifier}")
async def delete_sim_client(identifier: str):
    if not sim_client_cache.delete(identifier):
        raise HTTPException(status_code=404, detail="No existe ese cliente de simulación")
    return {"ok": True}


@router.delete("/clients")
async def clear_sim_clients():
    return {"ok": True, "removed": sim_client_cache.clear()}