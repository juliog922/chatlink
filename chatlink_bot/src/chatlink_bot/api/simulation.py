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
import os
import random
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy import delete, or_, select

from ..database import AsyncSessionPG, AsyncSessionSQL
from ..handlers import (
    _xlsx_from_cart, handle_new_email, handle_new_message, resolve_client_identity,
)
from ..logic.fsm import fsm
from ..logic.sim_clients import sim_client_cache, sim_history_cache
from ..models import (
    Chat, ClientHistorySync, ClientProductHistory, ConversationSession,
    EmailChat, MSClient, User,
)
from ..ai.qdrant import qdrant_service
from ..ai.rag import rag_service
from ..transport.outbox import sim_capture
from ..transport.whatsapp import WhatsAppMessage, whatsapp_transport

router = APIRouter()
logger = logging.getLogger("Simulation")

SIM_SALES_EMAIL = "sales@sim.com"
SIM_SALES_PHONE = "34600999001"

# Clients the simulator offers as actors. NOT the first five rows of SAGE, which
# is what it used before and which handed you whichever accounts happened to
# sort first — typically with no recent orders and nothing the bot may sell.
#
# These are picked for one property: recent orders containing articles with
# K_BOT = 0. Without that the history layer has nothing to resolve against and
# every test silently measures the catalog path only.
#
#   000025  VITAL LOOK SL          the long-standing reference. Careful: most of
#                                  its buying is 2012-2021, so it needs a wide
#                                  CLIENT_HISTORY_YEARS to be useful at all.
#   001764  PERFUMERIAS FACIAL     144 sellable, 260 orders, bought yesterday.
#   002652  QUE PUNTO ESTILISTAS   61 sellable vs only 99 excluded — the best
#                                  ratio available, so most of what it asks for
#                                  exercises the NORMAL path instead of the
#                                  "buys it but we may not sell it" branch.
#   090046  MARINA ROMANO ACOSTA   155 sellable, ordered yesterday. Email only:
#                                  the one that exercises the email channel.
#
# Override per install with SIM_CLIENT_CODES=000025,001764,…
SIM_CLIENT_CODES = [c.strip() for c in os.getenv(
    "SIM_CLIENT_CODES", "000025,001764,002652,090046").split(",") if c.strip()]

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

    try:  # SAGE clients worth testing against
        async with AsyncSessionSQL() as db:
            stmt = select(MSClient).where(MSClient.CodigoCliente.in_(SIM_CLIENT_CODES))
            rows = list((await db.execute(stmt)).scalars().all())
            if not rows:
                # The curated codes are not in this database (another install,
                # another company): fall back to whatever exists so the
                # simulator is never empty.
                rows = list((await db.execute(select(MSClient).limit(5))).scalars().all())
                logger.warning(f"[SIM] Ninguno de SIM_CLIENT_CODES existe aquí "
                               f"({SIM_CLIENT_CODES}); usando los primeros de SAGE.")
            order = {c: i for i, c in enumerate(SIM_CLIENT_CODES)}
            rows.sort(key=lambda c: order.get((c.CodigoCliente or "").strip(), 99))
            for c in rows:
                code = (c.CodigoCliente or "").strip()
                phone = (c.Telefono or c.Telefono2 or "").strip().replace(" ", "")
                email = (c.EMail1 or "").strip()
                label = f"[{code}] {c.Nombre}"
                if phone:
                    actors.append(SimActor(id=phone, name=label, type="client"))
                if email:
                    actors.append(SimActor(id=email, name=label, type="client",
                                           channel_pref="email"))
                if not phone and not email:
                    logger.info(f"[SIM] Cliente {code} sin teléfono ni email: no "
                                f"se puede simular tráfico suyo.")
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
    """Test clients, each with the size of its fake order history."""
    clients = sim_client_cache.list()
    for c in clients:
        c["fake_orders"] = sim_history_cache.count(c.get("code", ""))
    return {"clients": clients}


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
    # Optional one-shot: give the new client a purchase history immediately, so
    # the order-history resolution layer is testable the moment it exists.
    seeded = 0
    try:
        n = int(payload.get("seed_orders") or 0)
    except Exception:
        n = 0
    if n > 0:
        products = await _fabricate_history(n=n, seed=payload.get("seed"))
        seeded = sim_history_cache.set(entry["code"], products)
    return {"ok": True, "client": entry, "fake_orders": seeded}


@router.delete("/clients/{identifier}")
async def delete_sim_client(identifier: str):
    if not sim_client_cache.delete(identifier):
        raise HTTPException(status_code=404, detail="No existe ese cliente de simulación")
    return {"ok": True}


@router.delete("/clients")
async def clear_sim_clients():
    return {"ok": True, "removed": sim_client_cache.clear()}


# --------------------------------------------------------------------------
# Fake ORDER HISTORY for test clients.
#
# A test client resolves to CodigoCliente "TEST-xxxxxx", which does not exist
# in SAGE — so its order history is empty and the history-resolution layer
# cannot be exercised without a real recurring client. These endpoints give it
# a plausible history built from REAL catalog products (real codes, real
# descriptions, real brands, sampled from the indexed catalog) with fabricated
# order statistics. Memory only; a restart clears it.
#
# Products come from the Qdrant payloads, which are exactly the rows the
# ingestion considered bot-sellable (K_BOT = 0) — so a simulated history can
# never contain something the bot is not allowed to sell.
# --------------------------------------------------------------------------

class SimOrderSeedRequest(BaseModel):
    n: int = 12                      # how many distinct articles to invent
    queries: List[str] = []          # craft a scenario: search terms, top hits win
    codes: List[str] = []            # or pin exact article codes
    seed: Optional[int] = None       # reproducible fabrication
    replace: bool = True             # False = merge into the existing history


async def _sample_catalog(n: int) -> List[Dict[str, Any]]:
    """Random real products. Prefers the in-memory BM25 doc map (instant); falls
    back to a bounded Qdrant scroll so this stays fast on a cold index."""
    pool = list(rag_service.documents_map.values())
    if not pool:
        try:
            await qdrant_service.ensure_ready()
            points, _ = await qdrant_service.client.scroll(
                collection_name=qdrant_service.collection,
                limit=max(256, n * 20), with_payload=True, with_vectors=False)
            pool = [dict(p.payload) for p in points if p.payload]
        except Exception as e:
            logger.warning(f"[SIM] Could not sample the catalog: {e}")
            return []
    return pool


def _payload_to_row(p: Dict[str, Any], rnd: random.Random, span_days: int) -> Dict[str, Any]:
    """One fabricated history row for a real catalog product.

    The statistics are what the matcher's tie-breaks actually read: `veces`
    (how often it is ordered) and recency. Weighted so a few staples dominate
    and the long tail is ordered once or twice, like a real account.
    """
    veces = rnd.choices([1, 2, 3, 5, 8, 14, 21], weights=[30, 24, 18, 13, 8, 5, 2])[0]
    qty = rnd.choices([1, 2, 3, 6, 12, 24], weights=[34, 22, 16, 16, 8, 4])[0]
    last_offset = rnd.randint(1, max(2, span_days // 3))
    first_offset = rnd.randint(last_offset, span_days) if veces > 1 else last_offset
    now = datetime.now()
    code = str(p.get("CodigoArticulo") or "").strip()
    return {
        "codigo": code,
        "nombre": (p.get("DescripcionArticulo") or "").strip(),
        "nombres_historicos": [x for x in [(p.get("Descripcion2Articulo") or "").strip()] if x],
        "marca": (p.get("MarcaProducto") or "").strip(),
        # Populated on a minority of rows, mirroring how sparsely SAGE's
        # CodigodelCliente is actually filled in.
        "ref_cliente": f"REF{rnd.randint(100, 999)}" if rnd.random() < 0.15 else "",
        "veces": veces,
        "qty_moda": qty,
        "unidades_total": qty * veces,
        "primera_fecha": now - timedelta(days=first_offset),
        "ultima_fecha": now - timedelta(days=last_offset),
        "bot_visible": True,
    }


async def _fabricate_history(n: int = 12, queries: Optional[List[str]] = None,
                             codes: Optional[List[str]] = None,
                             seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Build a fake history: pinned codes, then scenario queries, then random fill."""
    rnd = random.Random(seed)
    span_days = int(float(os.getenv("CLIENT_HISTORY_YEARS", "5")) * 365)
    picked: Dict[str, Dict[str, Any]] = {}

    for code in (codes or []):
        code = str(code or "").strip()
        if not code:
            continue
        try:
            hits = await qdrant_service.search_by_exact_code(code)
            for h in hits[:1]:
                if h.payload:
                    picked[code] = dict(h.payload)
        except Exception as e:
            logger.warning(f"[SIM] Could not pin code {code}: {e}")

    # Scenario queries go through the REAL retriever, so "champu coco" seeds the
    # several near-identical products that make the ambiguous case reproducible.
    for q in (queries or []):
        q = str(q or "").strip()
        if not q:
            continue
        try:
            for h in await rag_service.retrieve(q, top_k=3):
                c = str(h.get("CodigoArticulo") or "").strip()
                if c:
                    picked.setdefault(c, h)
        except Exception as e:
            logger.warning(f"[SIM] Query '{q}' failed while seeding: {e}")

    if len(picked) < n:
        pool = await _sample_catalog(n - len(picked))
        rnd.shuffle(pool)
        for p in pool:
            c = str(p.get("CodigoArticulo") or "").strip()
            if c and c not in picked:
                picked[c] = p
            if len(picked) >= n:
                break

    return [_payload_to_row(p, rnd, span_days) for p in picked.values()]


def _sim_code_or_404(identifier: str) -> str:
    entry = sim_client_cache.get(identifier)
    if not entry:
        raise HTTPException(status_code=404, detail="No existe ese cliente de simulación")
    return entry["code"]


@router.post("/clients/{identifier}/orders", tags=["Simulation"])
async def seed_sim_orders(identifier: str, req: SimOrderSeedRequest):
    """Give a test client a fake purchase history of real catalog products."""
    code = _sim_code_or_404(identifier)
    products = await _fabricate_history(n=req.n, queries=req.queries,
                                        codes=req.codes, seed=req.seed)
    if not products:
        raise HTTPException(
            status_code=503,
            detail="No se pudo muestrear el catálogo (¿Qdrant vacío o sin ingestar?)")
    total = (sim_history_cache.set(code, products) if req.replace
             else sim_history_cache.extend(code, products))
    logger.info(f"[SIM] Fake history for {identifier} ({code}): "
                f"{len(products)} articles, {total} total.")
    return {"ok": True, "codigo_cliente": code, "added": len(products), "total": total,
            "productos": [_row_out(r) for r in products]}


@router.get("/clients/{identifier}/orders", tags=["Simulation"])
async def get_sim_orders(identifier: str):
    """The fake history currently attached to a test client."""
    code = _sim_code_or_404(identifier)
    rows = sim_history_cache.get(code)
    return {"codigo_cliente": code, "existe": rows is not None,
            "total": len(rows or []),
            "productos": [_row_out(r) for r in (rows or [])]}


@router.delete("/clients/{identifier}/orders", tags=["Simulation"])
async def clear_sim_orders(identifier: str):
    """Remove the fake history (the client itself stays)."""
    code = _sim_code_or_404(identifier)
    return {"ok": sim_history_cache.delete(code), "codigo_cliente": code}


def _row_out(r: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-safe view of a fabricated row (datetimes -> ISO)."""
    def iso(v):
        return v.isoformat() if isinstance(v, datetime) else v
    return {**r, "primera_fecha": iso(r.get("primera_fecha")),
            "ultima_fecha": iso(r.get("ultima_fecha"))}


# --------------------------------------------------------------------------
# SELF-CHAT history.
#
# In a self-chat the "client" is the salesman's OWN number, which is not in
# SAGE. Two things then happen, and neither yields a usable client code:
#
#   * the ingestion gatekeeper synthesises _fake_client() -> "FAKE001", and
#   * handle_ai_trigger only does that when is_simulation is True, so a REAL
#     full-mode self-chat from a linked device ends up with client_obj = None
#     and greets you as "Cliente".
#
# The fix needs no handler change: register that number in the simulation
# client cache, which every client lookup checks BEFORE SAGE. The number then
# resolves to a stable TEST-xxxxxx code with a proper name, in simulation AND
# in real full mode — and that code is what the fake history hangs off.
#
# FAKE001 is seeded too by default, because a self-chat started before the
# number was registered will already have landed on it.
# --------------------------------------------------------------------------

# Must match handlers._fake_client(). Changing it there means changing it here.
FAKE_CLIENT_CODE = "FAKE001"


class SimSelfChatSeedRequest(BaseModel):
    identifier: str                  # the salesman's OWN phone or email
    name: str = ""
    n: int = 12
    queries: List[str] = []
    codes: List[str] = []
    seed: Optional[int] = None
    replace: bool = True
    also_fake001: bool = True        # cover self-chats already on FAKE001


@router.post("/self-chat/orders", tags=["Simulation"])
async def seed_self_chat_orders(req: SimSelfChatSeedRequest):
    """Make a self-chat number resolvable AND give it a fake purchase history."""
    identifier = (req.identifier or "").strip()
    if not identifier:
        raise HTTPException(status_code=422, detail="'identifier' es obligatorio")

    entry = sim_client_cache.upsert(
        identifier,
        name=(req.name or "").strip() or f"Self-chat {identifier}",
        notes="self-chat: el comercial hablando con su propio número",
    )
    products = await _fabricate_history(n=req.n, queries=req.queries,
                                        codes=req.codes, seed=req.seed)
    if not products:
        raise HTTPException(
            status_code=503,
            detail="No se pudo muestrear el catálogo (¿Qdrant vacío o sin ingestar?)")

    codes_seeded = [entry["code"]]
    total = (sim_history_cache.set(entry["code"], products) if req.replace
             else sim_history_cache.extend(entry["code"], products))
    if req.also_fake001:
        (sim_history_cache.set if req.replace else sim_history_cache.extend)(
            FAKE_CLIENT_CODE, products)
        codes_seeded.append(FAKE_CLIENT_CODE)

    logger.info(f"[SIM] Self-chat history for {identifier}: {len(products)} articles "
                f"on {', '.join(codes_seeded)}.")
    return {"ok": True, "client": entry, "codigos": codes_seeded,
            "added": len(products), "total": total,
            "productos": [_row_out(r) for r in products]}


@router.get("/self-chat/orders", tags=["Simulation"])
async def get_self_chat_orders(identifier: str = ""):
    """State of the self-chat history, for the console."""
    entry = sim_client_cache.get(identifier) if identifier else None
    return {
        "registrado": entry is not None,
        "client": entry,
        "codigo_cliente": (entry or {}).get("code"),
        "n_propio": sim_history_cache.count((entry or {}).get("code", "")),
        "n_fake001": sim_history_cache.count(FAKE_CLIENT_CODE),
    }


@router.delete("/self-chat/orders", tags=["Simulation"])
async def clear_self_chat_orders(identifier: str = ""):
    """Clear the self-chat history (and FAKE001). The client entry stays."""
    entry = sim_client_cache.get(identifier) if identifier else None
    cleared = []
    if entry and sim_history_cache.delete(entry["code"]):
        cleared.append(entry["code"])
    if sim_history_cache.delete(FAKE_CLIENT_CODE):
        cleared.append(FAKE_CLIENT_CODE)
    return {"ok": True, "borrados": cleared}

# --------------------------------------------------------------------------
# RESET of simulation data.
#
# Testing leaves a trail: chat rows, a ConversationSession carrying a stale
# cart and an unresolved queue, a cached order history. "Start over" must not
# mean wiping the database, because real traffic lives in the same tables — so
# this deletes ONLY what belongs to the simulation clients, identified by their
# registered identifiers and by the TEST- / FAKE001 client codes.
#
# What it never touches: real clients' chats and sessions, SAGE (read-only
# everywhere), and by default the test client entries themselves — you almost
# always want the client back, just clean.
# --------------------------------------------------------------------------

class SimResetRequest(BaseModel):
    identifiers: List[str] = []      # empty = every registered test client
    incluir_fake001: bool = True     # unregistered self-chats and mock clients
    borrar_clientes: bool = False    # also drop the test client entries
    borrar_chats: bool = True
    borrar_sesiones: bool = True
    borrar_historial: bool = True


@router.post("/reset", tags=["Simulation"])
async def reset_simulation_data(req: SimResetRequest):
    """Wipe the traces of the simulation clients, leaving real data alone."""
    entries = sim_client_cache.list()
    if req.identifiers:
        wanted = {_norm_ident(i) for i in req.identifiers}
        entries = [e for e in entries if _norm_ident(e["identifier"]) in wanted]
    identifiers = [e["identifier"] for e in entries]
    codes = [e["code"] for e in entries if e.get("code")]
    if req.incluir_fake001:
        codes.append(FAKE_CLIENT_CODE)

    if not identifiers and not req.incluir_fake001:
        raise HTTPException(404, "No matching simulation clients.")

    removed: Dict[str, int] = {"chats": 0, "emails": 0, "sesiones": 0,
                               "historial_filas": 0, "historial_sync": 0,
                               "historial_memoria": 0, "clientes": 0}

    async with AsyncSessionPG() as db:
        if identifiers:
            if req.borrar_chats:
                for model, key in ((Chat, "chats"), (EmailChat, "emails")):
                    # A test identifier can appear as either participant: as the
                    # client in a normal simulation, as BOTH in a self-chat.
                    res = await db.execute(
                        delete(model).where(or_(model.client.in_(identifiers),
                                                model.chat_id.in_(identifiers))))
                    removed[key] += res.rowcount or 0
            if req.borrar_sesiones:
                res = await db.execute(
                    delete(ConversationSession)
                    .where(ConversationSession.client_id.in_(identifiers)))
                removed["sesiones"] += res.rowcount or 0
        if req.borrar_historial and codes:
            res = await db.execute(
                delete(ClientProductHistory)
                .where(ClientProductHistory.codigo_cliente.in_(codes)))
            removed["historial_filas"] += res.rowcount or 0
            res = await db.execute(
                delete(ClientHistorySync)
                .where(ClientHistorySync.codigo_cliente.in_(codes)))
            removed["historial_sync"] += res.rowcount or 0
        await db.commit()

    if req.borrar_historial:
        for c in codes:
            if sim_history_cache.delete(c):
                removed["historial_memoria"] += 1

    # The FSM holds in-memory debounce state per conversation; dropping the rows
    # underneath it would otherwise leave a timer pointing at a session that no
    # longer exists.
    for ident in identifiers:
        for ch in ("whatsapp", "email"):
            await fsm.on_user_message(ch, ident, SIM_SALES_PHONE if ch == "whatsapp" else SIM_SALES_EMAIL)

    if req.borrar_clientes:
        for ident in identifiers:
            if sim_client_cache.delete(ident):
                removed["clientes"] += 1

    logger.info(f"[SIM] Reset for {identifiers or ['(solo FAKE001)']} -> {removed}")
    return {"ok": True, "identificadores": identifiers, "codigos": codes,
            "borrado": removed}


def _norm_ident(value: str) -> str:
    v = (value or "").strip()
    return v.lower() if "@" in v else re.sub(r"\D", "", v)


class SimClearConversationRequest(BaseModel):
    """One conversation, addressed exactly."""
    channel: str                          # whatsapp | email
    client: str
    user: str = ""                        # defaults to the sim salesman
    borrar_historial_cache: bool = False  # force a fresh SAGE re-read next turn


@router.post("/conversation/clear", tags=["Simulation"])
async def clear_one_conversation(req: SimClearConversationRequest):
    """Erase ONE conversation: its messages and its persisted session state.

    Separate from /reset on purpose. /reset is scoped to the registered test
    clients; this is scoped to a single (channel, client, salesman) triple, so
    it also works when the simulation drives a REAL SAGE client — which is the
    common case, and the one /reset cannot reach.

    Deleting the session is what actually lets a retest start clean: the cart,
    the unresolved queue and the `estado="propuesto"` grounding carriers all
    live there, and a stale one is why the same code keeps coming back.
    """
    channel = (req.channel or "").strip()
    client = (req.client or "").strip()
    if channel not in ("whatsapp", "email") or not client:
        raise HTTPException(422, "'channel' (whatsapp|email) y 'client' son obligatorios")
    user = (req.user or "").strip() or (SIM_SALES_PHONE if channel == "whatsapp" else SIM_SALES_EMAIL)

    model = Chat if channel == "whatsapp" else EmailChat
    removed = {"mensajes": 0, "sesiones": 0, "historial_filas": 0}
    async with AsyncSessionPG() as db:
        res = await db.execute(delete(model).where(model.client == client, model.user == user))
        removed["mensajes"] = res.rowcount or 0
        res = await db.execute(delete(ConversationSession).where(
            ConversationSession.channel == channel,
            ConversationSession.client_id == client,
            ConversationSession.user_id == user))
        removed["sesiones"] = res.rowcount or 0
        if req.borrar_historial_cache:
            # Resolve the identifier to its CodigoCliente so the cached order
            # history is dropped too; the next turn re-reads SAGE.
            ident = await resolve_client_identity(client)
            if ident.code:
                r2 = await db.execute(delete(ClientProductHistory).where(
                    ClientProductHistory.codigo_cliente == ident.code))
                removed["historial_filas"] = r2.rowcount or 0
                await db.execute(delete(ClientHistorySync).where(
                    ClientHistorySync.codigo_cliente == ident.code))
                sim_history_cache.delete(ident.code)
        await db.commit()

    # Drop the in-memory debounce state, or a timer would keep pointing at a
    # session row that no longer exists.
    await fsm.on_user_message(channel, client, user)
    sim_capture.drain((channel, client, user))
    logger.info(f"[SIM] Cleared conversation {channel}/{client}/{user} -> {removed}")
    return {"ok": True, "channel": channel, "client": client, "user": user,
            "borrado": removed}


@router.get("/cart/xlsx", tags=["Simulation"])
async def download_cart_xlsx(channel: str, client: str, user: str = ""):
    """The current cart as the SAME xlsx the salesman receives on close.

    Deliberately built by handlers._xlsx_from_cart rather than assembled in the
    browser: an export that differs from the real deliverable is worse than no
    export, because it is trusted. One function, one format, one set of column
    names — codigo / cantidad.
    """
    user = (user or "").strip() or (SIM_SALES_PHONE if channel == "whatsapp" else SIM_SALES_EMAIL)
    async with AsyncSessionPG() as db:
        stmt = select(ConversationSession).where(
            ConversationSession.channel == channel,
            ConversationSession.client_id == client,
            ConversationSession.user_id == user)
        session = (await db.execute(stmt)).scalars().first()
    cart = (session.cart if session else None) or []
    if not cart:
        raise HTTPException(404, "El carrito está vacío.")
    data = _xlsx_from_cart(cart)
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="pedido_{client}.xlsx"'})