#chatlink_bot/src/chatlink_bot/api/routes.py
import os
import re
import time
from typing import Any, Dict, List, Optional
from datetime import datetime, date, timedelta, timezone
from decimal import Decimal

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import Integer, or_, text, union_all, literal, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from ..database import AsyncSessionPG, AsyncSessionSQL, get_db, pg_engine, sql_engine
from ..models import (
    User, UserCreate, UserResponse, Chat, EmailChat,
    ConversationSession, ClientProductHistory, ClientHistorySync,
    MSArticle, MSClient, MSOrderHeader, MSOrderLine,
)
from ..transport.whatsapp import whatsapp_transport
from ..transport.email import email_transport
from ..handlers import login_user, logout_user, resolve_client_identity
from ..ai.qdrant import qdrant_service
from ..ai.rag import SAGE_EMPRESA, client_history, rag_service, tokenize, variants
from ..ai import llm

router = APIRouter()
LOG_FILE_PATH = "app.log"

class SendMessageRequest(BaseModel):
    to: str
    text: Optional[str] = None
    from_jid: Optional[str] = None
    binary: Optional[bytes] = None
    filename: Optional[str] = None


class SendEmailRequest(BaseModel):
    to: str
    subject: str
    body: str


class DeviceIDRequest(BaseModel):
    jid: str


class DeviceListResponse(BaseModel):
    devices: List[str]


class ActionResponse(BaseModel):
    success: bool
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    details: dict


class ChatResponse(BaseModel):
    id: int
    chat_id: str
    user: str
    client: str
    message: Optional[str]
    direction: str
    # Without these the UI cannot distinguish the bot's replies from the
    # salesman's manual messages (they were stored but never serialized).
    is_bot: bool = False
    input_type: Optional[str] = None
    timestamp: datetime

    class Config:
        from_attributes = True

class UnifiedMessageResponse(BaseModel):
    channel: str  # "whatsapp" | "email"
    id: int
    chat_id: str
    user: str
    client: str
    message: Optional[str]
    direction: str
    input_type: str
    is_bot: bool
    timestamp: datetime

class LogEntry(BaseModel):
    line: str

class UserConnectionStatus(BaseModel):
    user_id: int
    name: str
    email: str
    phone: str
    whatsapp_connected: bool
    email_connected: bool

@router.get("/healthz", response_model=HealthResponse, tags=["System"])
async def health_check():
    status = "ok"
    details: dict = {}

    details["whatsapp_running"] = bool(getattr(whatsapp_transport, "is_running", False))
    details["email_mailboxes_active"] = getattr(email_transport, "active_mailboxes", [])

    try:
        async with pg_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        details["postgres"] = "ok"
    except Exception as e:
        details["postgres"] = f"error: {e}"
        status = "degraded"

    try:
        async with sql_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        details["sqlserver"] = "ok"
    except Exception as e:
        details["sqlserver"] = f"error: {e}"
        status = "degraded"

    q_health = await qdrant_service.health_check()
    details["qdrant"] = q_health
    if q_health["status"] != "ok":
        status = "degraded"

    c_health = await llm.health_check()
    details["cima"] = c_health
    details["cudara"] = c_health  # legacy alias; remove once nothing reads it
    if c_health.get("status") != "ok":
        status = "degraded"

    return {"status": status, "details": details}


@router.post("/users", response_model=UserResponse, status_code=201, tags=["Users"])
async def register_user(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == user_data.email))
    if result.scalars().first():
        raise HTTPException(400, "Email already registered")

    new_user = User(**user_data.model_dump())
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return new_user


@router.get("/users", response_model=List[UserResponse], tags=["Users"])
async def list_users(db: AsyncSession = Depends(get_db)):
    return (await db.execute(select(User))).scalars().all()


@router.post("/users/{user_id}/login", tags=["Users"])
async def api_login_user(user_id: int, db: AsyncSession = Depends(get_db)):
    user = (await db.execute(select(User).where(User.id == user_id))).scalars().first()
    if not user:
        raise HTTPException(404, "User not found")
    out = await login_user(user)
    if not out.get("success"):
        raise HTTPException(400, out.get("error", "login_failed"))
    return out


@router.post("/users/{user_id}/logout", tags=["Users"])
async def api_logout_user(user_id: int, db: AsyncSession = Depends(get_db)):
    user = (await db.execute(select(User).where(User.id == user_id))).scalars().first()
    if not user:
        raise HTTPException(404, "User not found")
    out = await logout_user(user)
    if not out.get("success"):
        raise HTTPException(400, out.get("error", "logout_failed"))
    return out

@router.delete("/users/{user_id}", tags=["Users"])
async def delete_user(user_id: int, db: AsyncSession = Depends(get_db)):
    user = (await db.execute(select(User).where(User.id == user_id))).scalars().first()
    if not user:
        raise HTTPException(404, "User not found")
    
    # Optional: cleanup transports
    if user.email:
        email_transport.stop_mailbox(user.email)
    
    await db.delete(user)
    await db.commit()
    return {"success": True}

@router.get("/chats", tags=["Messaging"])
async def get_chats(
    channel: str = Query("all", description="all | whatsapp | email"), 
    db: AsyncSession = Depends(get_db)
):
    """Returns unique chat IDs (conversations) ordered by most recent activity."""
    chats = []
    
    # 1. WhatsApp Conversations
    if channel in ("all", "whatsapp"):
        stmt = (
            select(Chat.chat_id, func.max(Chat.timestamp).label("last_ts"))
            .group_by(Chat.chat_id)
        )
        res = await db.execute(stmt)
        for row in res.all():
            chats.append({
                "chat_id": row.chat_id, 
                "channel": "whatsapp", 
                "last_activity": row.last_ts
            })
            
    # 2. Email Conversations
    if channel in ("all", "email"):
        stmt = (
            select(EmailChat.chat_id, func.max(EmailChat.timestamp).label("last_ts"))
            .group_by(EmailChat.chat_id)
        )
        res = await db.execute(stmt)
        for row in res.all():
            chats.append({
                "chat_id": row.chat_id, 
                "channel": "email", 
                "last_activity": row.last_ts
            })
            
    # Sort by activity desc
    chats.sort(key=lambda x: x["last_activity"], reverse=True)
    return chats

@router.get("/messages", response_model=List[ChatResponse], tags=["Messaging"])
async def list_messages(
    chat_id: Optional[str] = Query(None),
    phone: Optional[str] = Query(None),
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Chat).order_by(Chat.timestamp.desc()).limit(limit)
    if chat_id:
        stmt = stmt.where(Chat.chat_id == chat_id)
    if phone:
        stmt = stmt.where(or_(Chat.user == phone, Chat.client == phone))
    return (await db.execute(stmt)).scalars().all()


@router.post("/message", response_model=ActionResponse, tags=["Messaging"])
async def send_message(req: SendMessageRequest):
    resp = whatsapp_transport.send_message(
        to_phone=req.to,
        text=req.text,
        from_jid=req.from_jid,
        binary=req.binary,
        filename=req.filename,
    )
    return {"success": bool(resp.get("success")), "error": resp.get("error")}


@router.get("/emails", response_model=List[ChatResponse], tags=["Email"])
async def list_emails(
    chat_id: Optional[str] = Query(None),
    email: Optional[str] = Query(None),
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(EmailChat).order_by(EmailChat.timestamp.desc()).limit(limit)
    if chat_id:
        stmt = stmt.where(EmailChat.chat_id == chat_id)
    if email:
        stmt = stmt.where(or_(EmailChat.user == email, EmailChat.client == email))
    return (await db.execute(stmt)).scalars().all()


@router.post("/email/send", response_model=ActionResponse, tags=["Email"])
async def send_email_endpoint(req: SendEmailRequest, tasks: BackgroundTasks):
    def _send_wrap():
        ok, err = email_transport.send_email(to_email=req.to, subject=req.subject, body=req.body)
        if not ok:
            raise RuntimeError(err or "send_email_failed")

    tasks.add_task(_send_wrap)
    return {"success": True, "error": None}


@router.get("/devices", response_model=DeviceListResponse, tags=["Devices"])
async def list_devices():
    return {"devices": whatsapp_transport.list_devices()}


@router.post("/qdrant/ingest", response_model=ActionResponse, tags=["Qdrant"])
async def manual_ingest():
    total = await qdrant_service.ingest_products_from_sqlserver()
    await rag_service.initialize()
    return {"success": True, "error": f"upserted={total}"}

@router.get("/timeline", response_model=List[UnifiedMessageResponse], tags=["Messaging"])
async def timeline(
    day: Optional[str] = Query(None, description="YYYY-MM-DD in UTC"),
    limit: int = Query(100, ge=1, le=1000),
    channel: str = Query("all", description="all | whatsapp | email"),
    db: AsyncSession = Depends(get_db),
):
    # Day filter in UTC (timestamps are stored as timezone-aware UTC in your app)
    start_dt = end_dt = None
    if day:
        try:
            d = datetime.strptime(day, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(400, "Invalid day format. Use YYYY-MM-DD")
        start_dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        end_dt = start_dt + timedelta(days=1)

    def _apply_day(stmt, ts_col):
        if start_dt and end_dt:
            return stmt.where(ts_col >= start_dt, ts_col < end_dt)
        return stmt

    q_whatsapp = select(
        literal("whatsapp").label("channel"),
        Chat.id.label("id"),
        Chat.chat_id.label("chat_id"),
        Chat.user.label("user"),
        Chat.client.label("client"),
        Chat.message.label("message"),
        Chat.direction.label("direction"),
        Chat.input_type.label("input_type"),
        Chat.is_bot.label("is_bot"),
        Chat.timestamp.label("timestamp"),
    )
    q_whatsapp = _apply_day(q_whatsapp, Chat.timestamp)

    q_email = select(
        literal("email").label("channel"),
        EmailChat.id.label("id"),
        EmailChat.chat_id.label("chat_id"),
        EmailChat.user.label("user"),
        EmailChat.client.label("client"),
        EmailChat.message.label("message"),
        EmailChat.direction.label("direction"),
        EmailChat.input_type.label("input_type"),
        EmailChat.is_bot.label("is_bot"),
        EmailChat.timestamp.label("timestamp"),
    )
    q_email = _apply_day(q_email, EmailChat.timestamp)

    channel_l = (channel or "all").strip().lower()
    if channel_l == "whatsapp":
        union_q = q_whatsapp
    elif channel_l == "email":
        union_q = q_email
    else:
        union_q = union_all(q_whatsapp, q_email)

    if channel_l in ("whatsapp", "email"):
        stmt = union_q.order_by(text("timestamp DESC")).limit(limit)
        rows = (await db.execute(stmt)).mappings().all()
    else:
        subq = union_q.subquery("u")
        stmt = select(subq).order_by(subq.c.timestamp.desc()).limit(limit)
        rows = (await db.execute(stmt)).mappings().all()

    out = []
    for r in rows:
        item = dict(r)
        it = item.get("input_type")
        if hasattr(it, "value"):  # Enum -> string
            item["input_type"] = it.value
        out.append(item)

    return out

@router.get("/logs", response_model=List[LogEntry], tags=["System"])
async def get_logs(
    limit: int = 100, 
    level: str = Query(None, description="INFO, ERROR, WARNING"),
    search: str = Query(None)
):
    """
    Reads the last N lines from the log file, applying filters.
    """
    if not os.path.exists(LOG_FILE_PATH):
        return []

    lines = []
    # This is a simple implementation. For production huge logs, use `tail` or seek.
    # Reading entire file for simplicity in this context.
    try:
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
        
        # Reverse to get newest first
        all_lines.reverse()
        
        for line in all_lines:
            if len(lines) >= limit:
                break
            
            clean_line = line.strip()
            if not clean_line:
                continue

            # Level Filter
            if level:
                if f"[{level.upper()}]" not in clean_line:
                    continue
            
            # Text Filter
            if search:
                if search.lower() not in clean_line.lower():
                    continue

            lines.append(LogEntry(line=clean_line))
            
    except Exception as e:
        return [LogEntry(line=f"Error reading logs: {str(e)}")]

    return lines

@router.get("/connections", response_model=List[UserConnectionStatus], tags=["System"])
async def get_active_connections(db: AsyncSession = Depends(get_db)):
    """
    Returns the active connection status (WhatsApp and Email) for all real, enabled users.
    Excludes simulated accounts.
    """
    # 1. Fetch current active connections directly from the transports
    active_wa_jids = whatsapp_transport.list_devices()
    
    # Extract plain phone numbers from the JIDs (e.g., "34600111222:5@s.whatsapp.net" -> "34600111222")
    active_wa_phones = {jid.split("@")[0].split(":")[0] for jid in active_wa_jids if jid}
    
    active_emails = set(email_transport.active_mailboxes)
    
    # 2. Fetch all enabled users, explicitly filtering out the simulation domain
    stmt = select(User).where(
        User.email.not_like("sales@sim.com"), # Filter specific sim account
        User.email.not_like("admin@sim.com")  # Filter specific sim account
    )
    users = (await db.execute(stmt)).scalars().all()
    
    connections = []
    for u in users:
        clean_phone = (u.phone or "").strip()
        
        # Check WhatsApp: Is the user's phone number currently active in the transport?
        wa_connected = bool(clean_phone and clean_phone in active_wa_phones)
        
        # Check Email: Is the user's email actively being polled by a mailbox listener?
        email_connected = bool(u.email and u.email.strip().lower() in active_emails)
        
        connections.append(UserConnectionStatus(
            user_id=u.id,
            name=u.name,
            email=u.email,
            phone=u.phone,
            whatsapp_connected=wa_connected,
            email_connected=email_connected
        ))
        
    return connections

class SetGmailPasswordRequest(BaseModel):
    password: str

@router.post("/users/{user_id}/gmail-password", response_model=ActionResponse, tags=["Users"])
async def set_gmail_password(user_id: int, req: SetGmailPasswordRequest, db: AsyncSession = Depends(get_db)):
    user = (await db.execute(select(User).where(User.id == user_id))).scalars().first()
    if not user:
        return ActionResponse(success=False, error="User not found")
    
    ok = email_transport.set_app_password(user.email, req.password)
    if ok:
        return ActionResponse(success=True)


# =============================================================================
# ORDER-HISTORY DIAGNOSTICS
# =============================================================================
# The bot does not use these. They exist to answer "can we actually tell which
# product the client means?" BEFORE anything is wired into the conversation
# flow. Read-only against SAGE; the only thing they write is the Postgres cache.
#
#   curl 'localhost:8000/api/history/client?identifier=600123456'
#   curl 'localhost:8000/api/history/products?identifier=600123456&refresh=true'
#   curl -G 'localhost:8000/api/history/resolve' \
#        --data-urlencode 'identifier=600123456' \
#        --data-urlencode 'q=el shampoo de coco' \
#        --data-urlencode 'q=la mascarilla esa de 8 gr'

async def _identity_or_404(identifier: str):
    ident = await resolve_client_identity(identifier)
    if not ident.found:
        raise HTTPException(404, f"'{identifier}' does not resolve to any SAGE client.")
    return ident


def _identity_payload(ident) -> dict:
    return {
        "nombre": ident.name or None,
        "codigo_cliente": ident.code,
        "inequivoco": ident.unique,
        "historial_permitido": ident.history_allowed,
        "origen": ident.source,
        "tipo_coincidencia": ident.match_kind,
        "codigos_candidatos": ident.candidate_codes,
    }


def _resolution_summary(resolutions) -> dict:
    counts: dict = {}
    for r in resolutions:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    total = len(resolutions) or 1
    return {
        "total": len(resolutions),
        "por_veredicto": counts,
        "tasa_resueltos": round(counts.get("resolved", 0) / total, 3),
        "tasa_sin_senal": round(counts.get("none", 0) / total, 3),
    }


@router.get("/history/client", tags=["History"])
async def history_client(identifier: str = Query(..., description="Phone or email")):
    """Which CodigoCliente does this resolve to, and is it unambiguous?"""
    ident = await resolve_client_identity(identifier)
    return {"identificador": identifier, "encontrado": ident.found, **_identity_payload(ident)}


@router.get("/history/products", tags=["History"])
async def history_products(
    identifier: str = Query(..., description="Phone or email"),
    refresh: bool = Query(False, description="Force a re-read of SAGE"),
    limit: int = Query(50, ge=1, le=1000),
    incluir_excluidos: bool = Query(False, description="Also show K_BOT != 0"),
):
    """This client's product history, exactly as the matcher will see it."""
    ident = await _identity_or_404(identifier)
    if not ident.history_allowed:
        return {"identificador": identifier, **_identity_payload(ident),
                "aviso": "Ambiguous identity or no code: history stays disabled.",
                "productos": []}

    products, stats = await client_history.load(ident.code, force=refresh)
    visibles = [p for p in products if p.bot_visible]
    mostrar = products if incluir_excluidos else visibles
    return {
        "identificador": identifier,
        **_identity_payload(ident),
        "sincronizacion": stats.as_dict(),
        "totales": {
            "articulos_vendibles": len(visibles),
            "articulos_excluidos_k_bot": len(products) - len(visibles),
            "con_ref_cliente": sum(1 for p in products if p.ref_cliente),
        },
        "productos": [{
            "codigo": p.codigo,
            "nombre": p.nombre,
            "marca": p.marca,
            "nombres_historicos": p.nombres_historicos,
            "ref_cliente": p.ref_cliente or None,
            "veces": p.veces,
            "unidades_total": p.unidades_total,
            "qty_moda": p.qty_moda,
            "primera_fecha": p.primera_fecha.isoformat() if p.primera_fecha else None,
            "ultima_fecha": p.ultima_fecha.isoformat() if p.ultima_fecha else None,
            "vendible_por_bot": p.bot_visible,
        } for p in mostrar[:limit]],
    }


@router.get("/history/resolve", tags=["History"])
async def history_resolve(
    identifier: str = Query(..., description="Phone or email"),
    q: List[str] = Query(..., description="Informal mentions (repeatable)"),
    refresh: bool = Query(False),
):
    """Resolve informal mentions against the order history. The key endpoint."""
    ident = await _identity_or_404(identifier)
    if not ident.history_allowed:
        return {"identificador": identifier, **_identity_payload(ident),
                "aviso": "Ambiguous identity: history stays disabled.",
                "resoluciones": [{"consulta": x, "veredicto": "disabled"} for x in q]}

    resolutions, stats = await client_history.resolve_queries(ident.code, q, force=refresh)
    return {
        "identificador": identifier,
        **_identity_payload(ident),
        "sincronizacion": stats.as_dict(),
        "resumen": _resolution_summary(resolutions),
        "resoluciones": [r.as_dict() for r in resolutions],
    }


@router.post("/history/evaluate", tags=["History"])
async def history_evaluate(payload: dict = Body(...)):
    """Batch evaluation.

    Body:
      {"identifier": "600123456",
       "queries": ["shampoo coco", "rimel negro"],
       "esperado": {"shampoo coco": "KG001399"}}   # optional

    With `esperado` it measures real accuracy; without it, only the verdict
    distribution. Feed it real transcript wording to find where it breaks.
    """
    identifier = str(payload.get("identifier") or "").strip()
    queries = [str(x) for x in (payload.get("queries") or []) if str(x).strip()]
    expected = {str(k): str(v) for k, v in (payload.get("esperado") or {}).items()}
    if not identifier or not queries:
        raise HTTPException(400, "'identifier' and 'queries' are required.")

    ident = await _identity_or_404(identifier)
    if not ident.history_allowed:
        raise HTTPException(409, f"Ambiguous identity ({ident.describe()}); "
                                 "cannot evaluate the history layer.")

    resolutions, stats = await client_history.resolve_queries(ident.code, queries)
    detalle, aciertos, fallos, en_candidatos = [], 0, 0, 0
    for r in resolutions:
        top = r.matches[0].codigo if r.matches else None
        esperado = expected.get(r.query)
        estado = None
        if esperado:
            if r.verdict == "resolved" and top == esperado:
                estado, aciertos = "acierto", aciertos + 1
            elif r.verdict == "resolved":
                estado, fallos = "falso_positivo", fallos + 1
            elif any(m.codigo == esperado for m in r.matches):
                estado, en_candidatos = "en_candidatos", en_candidatos + 1
            else:
                estado, fallos = "no_encontrado", fallos + 1
        detalle.append({**r.as_dict(), "esperado": esperado, "estado": estado,
                        "elegido": top})

    out = {
        "identificador": identifier,
        **_identity_payload(ident),
        "sincronizacion": stats.as_dict(),
        "resumen": _resolution_summary(resolutions),
        "detalle": detalle,
    }
    if expected:
        out["exactitud"] = {
            "aciertos": aciertos,
            "falsos_positivos": fallos,
            "solo_en_candidatos": en_candidatos,
            "precision_resueltos": round(aciertos / max(1, aciertos + fallos), 3),
            "cobertura": round(aciertos / max(1, len(expected)), 3),
        }
    return out
    return ActionResponse(success=False, error="Could not write to gmail_accounts.json")

# =============================================================================
# POSTGRES BROWSER + CATALOG INSPECTOR
# =============================================================================
# Read-only windows onto the bot's own state, so "why did Kapa do that?" can be
# answered from the console instead of from app.log. The one destructive action
# (resetting a conversation) is explicit and separate.

@router.get("/db/overview", tags=["State"])
async def db_overview(db: AsyncSession = Depends(get_db)):
    """Row counts per table + vector index size. The system at a glance."""
    tables = {
        "app_users": User,
        "chats": Chat,
        "email_chats": EmailChat,
        "conversation_sessions": ConversationSession,
        "client_product_history": ClientProductHistory,
        "client_history_sync": ClientHistorySync,
    }
    counts = {}
    for name, model in tables.items():
        try:
            counts[name] = int((await db.execute(
                select(func.count()).select_from(model))).scalar() or 0)
        except Exception as e:
            # A table that does not exist yet (fresh DB, or the history tables
            # before the first boot with the new models) must not 500 the view.
            counts[name] = None
            counts[f"{name}__error"] = str(e)[:120]

    # Conversation health: how many are mid-order right now.
    try:
        by_status = {
            str(row[0] or "IDLE"): int(row[1]) for row in (await db.execute(
                select(ConversationSession.order_status, func.count())
                .group_by(ConversationSession.order_status))).all()
        }
        open_convs = int((await db.execute(
            select(func.count()).select_from(ConversationSession)
            .where(ConversationSession.conv_open.is_(True)))).scalar() or 0)
        opted_out = int((await db.execute(
            select(func.count()).select_from(ConversationSession)
            .where(ConversationSession.bot_enabled.is_(False)))).scalar() or 0)
    except Exception:
        by_status, open_convs, opted_out = {}, 0, 0

    return {
        "tablas": counts,
        "conversaciones": {"por_estado": by_status, "abiertas": open_convs,
                           "opt_out": opted_out},
        "qdrant": {"puntos": await qdrant_service.get_collection_count(),
                   **(await qdrant_service.health_check())},
        "bm25_cargado": rag_service.bm25 is not None,
        "documentos_bm25": len(rag_service.corpus_ids),
    }


def _session_payload(s: ConversationSession, full: bool = False) -> dict:
    cart = list(s.cart or [])
    open_items = list(getattr(s, "open_items", None) or [])
    out = {
        "id": s.id,
        "channel": s.channel,
        "client_id": s.client_id,
        "user_id": s.user_id,
        "order_status": s.order_status or "IDLE",
        "conv_open": bool(s.conv_open),
        "bot_enabled": bool(s.bot_enabled),
        "guide_shown": bool(getattr(s, "guide_shown", False)),
        "n_cart": len(cart),
        "n_open_items": len(open_items),
        "n_last_closed": len(s.last_closed_cart or []),
        "summary": (s.summary or ""),
        "bot_introduced_at": s.bot_introduced_at.isoformat() if s.bot_introduced_at else None,
        "last_client_msg_at": s.last_client_msg_at.isoformat() if s.last_client_msg_at else None,
        "updated_at": s.updated_at.isoformat() if s.updated_at else None,
    }
    if full:
        out.update({"cart": cart, "open_items": open_items,
                    "last_closed_cart": list(s.last_closed_cart or [])})
    return out


@router.get("/db/sessions", tags=["State"])
async def db_sessions(
    channel: str = Query("all", description="all | whatsapp | email"),
    status: str = Query("all", description="all | IDLE | BUILDING | CLOSED"),
    q: Optional[str] = Query(None, description="Substring of client or salesman id"),
    solo_abiertas: bool = Query(False),
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """The ConversationSession table — the bot's persisted per-conversation state."""
    stmt = select(ConversationSession).order_by(ConversationSession.updated_at.desc())
    if channel in ("whatsapp", "email"):
        stmt = stmt.where(ConversationSession.channel == channel)
    if status in ("IDLE", "BUILDING", "CLOSED"):
        stmt = stmt.where(ConversationSession.order_status == status)
    if solo_abiertas:
        stmt = stmt.where(ConversationSession.conv_open.is_(True))
    if q:
        like = f"%{q.strip()}%"
        stmt = stmt.where(or_(ConversationSession.client_id.like(like),
                              ConversationSession.user_id.like(like)))
    rows = list((await db.execute(stmt.limit(limit))).scalars().all())
    return {"total": len(rows), "sesiones": [_session_payload(s) for s in rows]}


@router.get("/db/sessions/{session_id}", tags=["State"])
async def db_session_detail(session_id: int, db: AsyncSession = Depends(get_db)):
    """One conversation in full: cart lines, the unresolved queue, memory."""
    s = (await db.execute(select(ConversationSession)
                          .where(ConversationSession.id == session_id))).scalars().first()
    if not s:
        raise HTTPException(404, "Session not found")
    payload = _session_payload(s, full=True)

    # Resolve cart codes to names so the cart is readable, not just codes.
    nombres = {}
    for item in payload["cart"] + payload["last_closed_cart"]:
        code = str(item.get("code") or "").strip()
        if not code or code in nombres:
            continue
        try:
            hits = await qdrant_service.search_by_exact_code(code)
            if hits and hits[0].payload:
                nombres[code] = {
                    "nombre": hits[0].payload.get("DescripcionArticulo") or "",
                    "marca": hits[0].payload.get("MarcaProducto") or "",
                }
        except Exception:
            pass
    payload["nombres_articulos"] = nombres
    return payload


@router.delete("/db/sessions/{session_id}", response_model=ActionResponse, tags=["State"])
async def db_session_reset(session_id: int, db: AsyncSession = Depends(get_db)):
    """DESTRUCTIVE: wipe a conversation's state.

    For a conversation stuck mid-order (a cart that will not close, a queue that
    keeps re-asking). The message history in chats/email_chats is untouched —
    only the session state is cleared, so the next client message starts a fresh
    conversation with a re-introduction.
    """
    s = (await db.execute(select(ConversationSession)
                          .where(ConversationSession.id == session_id))).scalars().first()
    if not s:
        return ActionResponse(success=False, error="Session not found")
    s.order_status, s.cart, s.summary = "IDLE", [], ""
    s.last_closed_cart = []
    if hasattr(type(s), "open_items"):
        s.open_items = []
    if hasattr(type(s), "guide_shown"):
        s.guide_shown = False
    s.conv_open, s.bot_enabled = False, True
    s.bot_introduced_at = None
    await db.commit()
    return ActionResponse(success=True)


@router.get("/db/catalog/search", tags=["State"])
async def db_catalog_search(
    q: str = Query(..., description="Exactly what a client would type"),
    top_k: int = Query(5, ge=1, le=25),
):
    """Run the REAL retrieval path and show what the agent would receive.

    This is the "why did it return that?" view: the tokens the shared
    normalizer produced, their cross-language variants, and per-hit
    covers_query / query_uncovered — the two signals _triage_results uses to
    decide between committing an item and asking about it.
    """
    toks = tokenize(q)
    try:
        hits = await rag_service.retrieve(q, top_k=top_k)
    except Exception as e:
        raise HTTPException(500, f"Retrieval failed: {e}")
    return {
        "consulta": q,
        "tokens": toks,
        "variantes": {t: sorted(variants(t)) for t in toks},
        "sin_coincidencia": list((hits[0].get("query_uncovered") or []) if hits else []),
        "bm25_cargado": rag_service.bm25 is not None,
        "resultados": [{
            "codigo": h.get("CodigoArticulo"),
            "nombre": h.get("DescripcionArticulo") or "",
            "nombre2": h.get("Descripcion2Articulo") or "",
            "linea": h.get("DescripcionLinea") or "",
            "marca": h.get("MarcaProducto") or "",
            "score": round(float(h.get("relevance_score") or 0.0), 4),
            "cubre_consulta": bool(h.get("covers_query")),
            "no_cubre": list(h.get("query_uncovered") or []),
        } for h in hits],
    }


@router.get("/history/clients", tags=["History"])
async def history_clients(
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """Which clients have a cached order history, and how that sync went.

    The column to watch is `tolerant_join`: if it is true anywhere, SeriePedido
    is inconsistent between CabeceraPedidoCliente and LineasPedidoCliente. And
    `n_excluded` much larger than `n_products` means the K_BOT assumption is
    probably inverted for this catalog.
    """
    rows = list((await db.execute(
        select(ClientHistorySync)
        .order_by(ClientHistorySync.refreshed_at.desc().nullslast())
        .limit(limit))).scalars().all())
    return {"total": len(rows), "clientes": [{
        "codigo_cliente": r.codigo_cliente,
        "status": r.status,
        "n_products": r.n_products or 0,
        "n_lines": r.n_lines or 0,
        "n_orders": r.n_orders or 0,
        "n_excluded": r.n_excluded or 0,
        "n_ref_cliente": r.n_ref_cliente or 0,
        "tolerant_join": bool(r.tolerant_join),
        "error": r.error or "",
        "refreshed_at": r.refreshed_at.isoformat() if r.refreshed_at else None,
    } for r in rows]}


# =============================================================================
# DATABASE EXPLORER
# =============================================================================
# Browsing shaped around what these tables MEAN, not around rows: chats render
# as conversations, orders as a header plus its lines, products as a catalog.
# The raw-SQL console at the end is the escape hatch for everything else.

# ---------------------------------------------------------------- chat threads
@router.get("/db/chats/threads", tags=["Explorer"])
async def db_chat_threads(
    channel: str = Query("all", description="all | whatsapp | email"),
    q: Optional[str] = Query(None, description="Substring of either participant"),
    limit: int = Query(200, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
):
    """Conversation threads, keyed by (channel, client, salesman).

    /api/chats groups by chat_id alone, which merges the same client talking to
    two different salesmen into one row. A thread is the triple, because that is
    the unit the bot actually reasons about (and the key of a session).
    """
    threads = []
    sources = []
    if channel in ("all", "whatsapp"):
        sources.append(("whatsapp", Chat))
    if channel in ("all", "email"):
        sources.append(("email", EmailChat))

    for name, model in sources:
        stmt = (
            select(model.client, model.user,
                   func.count().label("n"),
                   func.max(model.timestamp).label("last_ts"),
                   func.sum(func.cast(model.is_bot, Integer)).label("n_bot"))
            .group_by(model.client, model.user)
            .order_by(func.max(model.timestamp).desc())
            .limit(limit)
        )
        if q:
            like = f"%{q.strip()}%"
            stmt = stmt.where(or_(model.client.like(like), model.user.like(like)))
        for row in (await db.execute(stmt)).all():
            threads.append({
                "channel": name, "client": row.client, "user": row.user,
                "n_messages": int(row.n or 0), "n_bot": int(row.n_bot or 0),
                "last_activity": row.last_ts.isoformat() if row.last_ts else None,
            })
    threads.sort(key=lambda t: t["last_activity"] or "", reverse=True)
    return {"total": len(threads), "threads": threads[:limit]}


@router.get("/db/chats/thread", tags=["Explorer"])
async def db_chat_thread(
    channel: str = Query(..., description="whatsapp | email"),
    client: str = Query(...),
    user: str = Query(...),
    limit: int = Query(300, ge=1, le=2000),
    db: AsyncSession = Depends(get_db),
):
    """Every message of one thread, oldest first — the chat view."""
    model = Chat if channel == "whatsapp" else EmailChat
    stmt = (select(model)
            .where(model.client == client, model.user == user)
            .order_by(model.timestamp.desc()).limit(limit))
    rows = list((await db.execute(stmt)).scalars().all())
    rows.reverse()
    return {"channel": channel, "client": client, "user": user, "total": len(rows),
            "mensajes": [{
                "id": r.id,
                "message": r.message or "",
                "direction": r.direction,
                "is_bot": bool(r.is_bot),
                "input_type": r.input_type.value if hasattr(r.input_type, "value") else str(r.input_type),
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            } for r in rows]}


# -------------------------------------------------------------- SAGE: clients
@router.get("/sage/clients", tags=["Explorer"])
async def sage_clients(
    q: Optional[str] = Query(None, description="Name, code, phone, email or CIF"),
    limit: int = Query(50, ge=1, le=500),
):
    """Search SAGE Clientes. The starting point for inspecting a real client."""
    stmt = select(MSClient).order_by(MSClient.CodigoCliente).limit(limit)
    if q and q.strip():
        like = f"%{q.strip()}%"
        stmt = stmt.where(or_(
            MSClient.Nombre.like(like), MSClient.CodigoCliente.like(like),
            MSClient.Telefono.like(like), MSClient.Telefono2.like(like),
            MSClient.Telefono3.like(like), MSClient.EMail1.like(like),
            MSClient.EMail2.like(like), MSClient.CifDni.like(like),
        ))
    try:
        async with AsyncSessionSQL() as s:
            rows = list((await s.execute(stmt)).scalars().all())
    except Exception as e:
        raise HTTPException(502, f"SAGE unreachable: {e}")
    return {"total": len(rows), "clientes": [{
        "codigo_cliente": (c.CodigoCliente or "").strip(),
        "empresa": c.CodigoEmpresa,
        "nombre": (c.Nombre or "").strip(),
        "telefonos": [t for t in [(c.Telefono or "").strip(), (c.Telefono2 or "").strip(),
                                  (c.Telefono3 or "").strip()] if t],
        "emails": [e for e in [(c.EMail1 or "").strip(), (c.EMail2 or "").strip()] if e],
        "municipio": (c.Municipio or "").strip(),
        "provincia": (c.Provincia or "").strip(),
        "cif": (c.CifDni or "").strip(),
    } for c in rows]}


# --------------------------------------------------------------- SAGE: orders
@router.get("/sage/orders", tags=["Explorer"])
async def sage_orders(
    codigo_cliente: Optional[str] = Query(None),
    q: Optional[str] = Query(None, description="Substring of the company name"),
    limit: int = Query(100, ge=1, le=500),
):
    """Order headers, most recent first. Past orders for ANY client, not just mocks."""
    stmt = (select(MSOrderHeader)
            .where(MSOrderHeader.CodigoEmpresa == SAGE_EMPRESA)
            .order_by(MSOrderHeader.FechaPedido.desc())
            .limit(limit))
    if codigo_cliente and codigo_cliente.strip():
        # Zero-padded in SAGE ('000139'): exact string, never an int cast.
        stmt = stmt.where(MSOrderHeader.CodigoCliente == codigo_cliente.strip())
    if q and q.strip():
        stmt = stmt.where(MSOrderHeader.RazonSocial.like(f"%{q.strip()}%"))
    try:
        async with AsyncSessionSQL() as s:
            rows = list((await s.execute(stmt)).scalars().all())
    except Exception as e:
        raise HTTPException(502, f"SAGE unreachable: {e}")
    return {"total": len(rows), "pedidos": [{
        "ejercicio": h.EjercicioPedido,
        "serie": (h.SeriePedido or ""),
        "numero": h.NumeroPedido,
        "fecha": h.FechaPedido.isoformat() if h.FechaPedido else None,
        "codigo_cliente": (h.CodigoCliente or "").strip(),
        "razon_social": (h.RazonSocial or "").strip(),
        "n_lineas": h.NumeroLineas or 0,
        "importe": float(h.ImporteLiquido) if h.ImporteLiquido is not None else None,
    } for h in rows]}


@router.get("/sage/orders/lines", tags=["Explorer"])
async def sage_order_lines(
    ejercicio: int = Query(...),
    numero: int = Query(...),
    serie: str = Query("", description="Often empty — it is a string, not NULL"),
):
    """The lines of one order.

    Serie travels as a query parameter and not in the path because it is
    routinely an EMPTY STRING in the real data, which a path segment cannot
    carry. The tolerant COALESCE retry mirrors the history sync: if the strict
    4-column match finds nothing, SeriePedido is inconsistent between the two
    tables and that is worth surfacing rather than showing an empty order.
    """
    async def _fetch(tolerant: bool):
        keys = [MSOrderLine.CodigoEmpresa == SAGE_EMPRESA,
                MSOrderLine.EjercicioPedido == ejercicio,
                MSOrderLine.NumeroPedido == numero]
        keys.append(func.coalesce(MSOrderLine.SeriePedido, "") == (serie or "")
                    if tolerant else MSOrderLine.SeriePedido == (serie or ""))
        stmt = (select(MSOrderLine).where(*keys)
                .order_by(MSOrderLine.Orden).limit(500))
        async with AsyncSessionSQL() as s:
            return list((await s.execute(stmt)).scalars().all())

    try:
        rows = await _fetch(False)
        tolerant = False
        if not rows:
            rows = await _fetch(True)
            tolerant = bool(rows)
    except Exception as e:
        raise HTTPException(502, f"SAGE unreachable: {e}")

    return {"ejercicio": ejercicio, "serie": serie, "numero": numero,
            "tolerant_join": tolerant, "total": len(rows), "lineas": [{
                "orden": l.Orden,
                "codigo": (l.CodigoArticulo or "").strip(),
                "descripcion": (l.DescripcionArticulo or "").strip(),
                "descripcion2": (l.Descripcion2Articulo or "").strip(),
                "ref_cliente": (l.CodigodelCliente or "").strip(),
                "unidades": float(l.UnidadesPedidas) if l.UnidadesPedidas is not None else 0,
                "precio": float(l.Precio) if l.Precio is not None else None,
                "importe": float(l.ImporteNeto) if l.ImporteNeto is not None else None,
            } for l in rows]}


# ------------------------------------------------------------- SAGE: products
@router.get("/sage/products", tags=["Explorer"])
async def sage_products(
    q: Optional[str] = Query(None, description="Substring of code or description"),
    marca: Optional[str] = Query(None),
    solo_bot: bool = Query(True, description="Only K_BOT = 0, what the bot may sell"),
    limit: int = Query(100, ge=1, le=1000),
):
    """The catalog itself — so mock conversations can name products that exist."""
    stmt = select(MSArticle).order_by(MSArticle.CodigoArticulo).limit(limit)
    if solo_bot:
        stmt = stmt.where(MSArticle.K_BOT == 0)
    if marca and marca.strip():
        stmt = stmt.where(MSArticle.MarcaProducto == marca.strip())
    if q and q.strip():
        like = f"%{q.strip()}%"
        stmt = stmt.where(or_(MSArticle.CodigoArticulo.like(like),
                              MSArticle.DescripcionArticulo.like(like),
                              MSArticle.Descripcion2Articulo.like(like)))
    try:
        async with AsyncSessionSQL() as s:
            rows = list((await s.execute(stmt)).scalars().all())
    except Exception as e:
        raise HTTPException(502, f"SAGE unreachable: {e}")
    seen, out = set(), []
    for a in rows:                      # dedupe by code, as the ingestion does
        code = (a.CodigoArticulo or "").strip()
        if not code or code in seen:
            continue
        seen.add(code)
        out.append({
            "codigo": code,
            "descripcion": (a.DescripcionArticulo or "").strip(),
            "descripcion2": (a.Descripcion2Articulo or "").strip(),
            "linea": (a.DescripcionLinea or "").strip(),
            "marca": (a.MarcaProducto or "").strip(),
            "precio": float(a.PrecioVenta) if a.PrecioVenta is not None else None,
            "k_bot": a.K_BOT,
        })
    return {"total": len(out), "productos": out}


@router.get("/sage/brands", tags=["Explorer"])
async def sage_brands(limit: int = Query(400, ge=1, le=2000)):
    """Distinct brands — the segmentation axis for disambiguating products."""
    try:
        async with AsyncSessionSQL() as s:
            rows = (await s.execute(
                select(MSArticle.MarcaProducto, func.count().label("n"))
                .where(MSArticle.K_BOT == 0, MSArticle.MarcaProducto.isnot(None),
                       MSArticle.MarcaProducto != "")
                .group_by(MSArticle.MarcaProducto)
                .order_by(func.count().desc()).limit(limit))).all()
    except Exception as e:
        raise HTTPException(502, f"SAGE unreachable: {e}")
    return {"marcas": [{"marca": (r[0] or "").strip(), "n": int(r[1])} for r in rows]}


# ------------------------------------------------------------- raw SQL console
# The escape hatch for everything the shaped views do not cover. Read-only by
# construction, in layers, because "it is only my own console" is exactly how
# an ops tool ends up truncating a production table:
#
#   1. comments are stripped first, so nothing hides behind -- or /* */
#   2. the statement must START with SELECT or WITH
#   3. one statement only: a ';' with anything after it is rejected
#   4. a whole-word blocklist of everything that writes
#   5. the query is WRAPPED in a subquery with a hard row cap — this is the
#      real guard, since it forces the input to be a single expression: a
#      smuggled second statement becomes a syntax error rather than a command
#   6. Postgres additionally runs in a READ ONLY transaction with a statement
#      timeout, and every session is rolled back regardless of outcome
#
# NOTE on SAGE: SQL Server has no per-transaction read-only mode here, so
# layers 1-5 are what stand between a typo and a write. Point the SQLSERVER_USER
# credentials at a db_datareader-only login — that is the guarantee that does
# not depend on this parser being clever enough.

_SQL_COMMENTS = re.compile(r"--[^\n]*|/\*.*?\*/", re.S)
_SQL_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|merge|"
    r"exec|execute|call|copy|into|backup|restore|shutdown|reconfigure|"
    r"sp_\w+|xp_\w+)\b", re.I)
SQL_MAX_ROWS = 500
SQL_TIMEOUT_MS = 10000


class SqlQueryRequest(BaseModel):
    source: str = "postgres"          # postgres | sage
    sql: str
    limit: int = 200


def _validate_select(sql: str) -> str:
    """Return the cleaned statement, or raise 400 explaining what was refused."""
    cleaned = _SQL_COMMENTS.sub(" ", sql or "").strip().rstrip(";").strip()
    if not cleaned:
        raise HTTPException(400, "Empty query.")
    if ";" in cleaned:
        raise HTTPException(400, "One statement at a time — ';' is not allowed mid-query.")
    if not re.match(r"^\s*(select|with)\b", cleaned, re.I):
        raise HTTPException(400, "Read-only console: the query must start with SELECT or WITH.")
    bad = _SQL_FORBIDDEN.search(cleaned)
    if bad:
        raise HTTPException(400, f"Refused: '{bad.group(0)}' can modify data. "
                                 f"This console is read-only.")
    return cleaned


def _jsonable(v: Any) -> Any:
    """Decimals, datetimes, UUIDs and bytes all appear in these tables."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, (bytes, bytearray, memoryview)):
        return f"<{len(bytes(v))} bytes>"
    return str(v)


@router.post("/db/query", tags=["Explorer"])
async def db_query(req: SqlQueryRequest):
    """Run one read-only SELECT against Postgres or SAGE."""
    cleaned = _validate_select(req.sql)
    n = max(1, min(int(req.limit or 200), SQL_MAX_ROWS))
    started = time.perf_counter()

    if req.source == "sage":
        # ORDER BY inside a subquery needs TOP on SQL Server, so a query that
        # sorts must carry its own TOP. Said plainly rather than silently
        # rewritten, because rewriting someone's SQL is worse than refusing.
        wrapped = f"SELECT TOP {n} * FROM (\n{cleaned}\n) AS _q"
        try:
            async with AsyncSessionSQL() as s:
                res = await s.execute(text(wrapped))
                rows = [dict(r) for r in res.mappings().all()]
                await s.rollback()
        except Exception as e:
            msg = str(e)
            if "ORDER BY" in msg.upper():
                msg += ("  ·  On SQL Server an ORDER BY inside a subquery requires TOP: "
                        "write 'SELECT TOP 100 ... ORDER BY ...' in your own query.")
            raise HTTPException(400, msg[:800])
    else:
        wrapped = f"SELECT * FROM (\n{cleaned}\n) AS _q LIMIT {n}"
        try:
            async with AsyncSessionPG() as s:
                await s.execute(text("SET TRANSACTION READ ONLY"))
                await s.execute(text(f"SET LOCAL statement_timeout = {SQL_TIMEOUT_MS}"))
                res = await s.execute(text(wrapped))
                rows = [dict(r) for r in res.mappings().all()]
                await s.rollback()
        except Exception as e:
            raise HTTPException(400, str(e)[:800])

    rows = [{k: _jsonable(v) for k, v in r.items()} for r in rows]
    return {
        "source": req.source,
        "columnas": list(rows[0].keys()) if rows else [],
        "total": len(rows),
        "truncado": len(rows) >= n,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        "filas": rows,
    }


@router.get("/db/schema", tags=["Explorer"])
async def db_schema(source: str = Query("postgres", description="postgres | sage")):
    """Table and column names, so the console does not need guesswork."""
    try:
        if source == "sage":
            async with AsyncSessionSQL() as s:
                rows = (await s.execute(text(
                    "SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE "
                    "FROM INFORMATION_SCHEMA.COLUMNS "
                    "WHERE TABLE_NAME IN ('Clientes','Articulos'," 
                    "'CabeceraPedidoCliente','LineasPedidoCliente') "
                    "ORDER BY TABLE_NAME, ORDINAL_POSITION"))).all()
        else:
            async with AsyncSessionPG() as s:
                rows = (await s.execute(text(
                    "SELECT table_name, column_name, data_type "
                    "FROM information_schema.columns "
                    "WHERE table_schema = 'public' "
                    "ORDER BY table_name, ordinal_position"))).all()
    except Exception as e:
        raise HTTPException(502, f"Schema unavailable: {e}")
    tables: Dict[str, List[Dict[str, str]]] = {}
    for t, c, d in rows:
        tables.setdefault(t, []).append({"columna": c, "tipo": d})
    return {"source": source, "tablas": tables}