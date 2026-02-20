#chatlink_bot/src/chatlink_bot/api/routes.py
import os
from typing import List, Optional
from datetime import datetime, date, timedelta, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import or_, text, union_all, literal, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from ..database import get_db, pg_engine, sql_engine
from ..models import User, UserCreate, UserResponse, Chat, EmailChat
from ..transport.whatsapp import whatsapp_transport
from ..transport.email import email_transport
from ..handlers import login_user, logout_user
from ..ai.qdrant import qdrant_service
from ..ai.rag import rag_service
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
    details["cudara"] = c_health
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
        User.enabled == True,
        User.email.not_like("%@sim.com%")
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