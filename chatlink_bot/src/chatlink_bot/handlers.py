import asyncio
import logging
import os
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, or_, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from .database import AsyncSessionPG, AsyncSessionSQL
from .events import event_bus
from .models import Chat, EmailChat, InputType, MSClient, User
from .logic.fsm import fsm
from .transport.whatsapp import whatsapp_transport
from .transport.email import email_transport, SMTP_USER as ADMIN_SMTP_USER

from .ai.parsers import (
    extract_text_from_image_bytes_async,
    transcribe_audio_bytes_async,
    extract_text_from_document_bytes_async,
)
from .ai.rag import rag_service
from .ai.llm import (
    get_summary_state,
    set_summary_state,
    summarize_update_async,
    build_order_reply_async,
)

logger = logging.getLogger("Handlers")

COMMERCIAL_EMAIL = os.getenv("COMMERCIAL_EMAIL", "").strip()
GUARDRAIL_TEXT = os.getenv("BOT_GUARDRAIL_TEXT", "El comercial te contactará pronto.")
MAX_HISTORY = int(os.getenv("BOT_HISTORY_LIMIT", "60"))

# For admin command parsing when admin is stored in Postgres (not only ADMIN_WA_NUMBERS)
ADMIN_CMD_RE = re.compile(r"^\s*(login|logout)\b", re.IGNORECASE)
ADMIN_HELP_TEXT = (
    "🤖 *ChatLink Admin Help*\n\n"
    "Para gestionar tu acceso, utiliza los siguientes comandos:\n\n"
    "• *login*: Activa el monitoreo de WhatsApp y Email para tu cuenta.\n"
    "• *logout*: Desactiva el servicio y cierra las sesiones activas.\n"
)

# prevent concurrent AI work per conversation
_processing_locks: Dict[Tuple[str, str, str], asyncio.Lock] = {}

_admin_help_cache: Dict[str, datetime] = {}
_admin_command_cache: Dict[str, datetime] = {}

def _prune_caches():
    """Prevents memory leaks by capping caches to 1000 items."""
    if len(_admin_help_cache) > 1000:
        _admin_help_cache.clear()
    if len(_admin_command_cache) > 1000:
        _admin_command_cache.clear()
    if len(_processing_locks) > 5000:
        _processing_locks.clear()

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _norm_phone(p: str) -> str:
    p = (p or "").strip()
    p = re.sub(r"[^\d+]", "", p)
    # keep last 9 digits as loose matching
    digits = re.sub(r"\D", "", p)
    return digits[-9:] if len(digits) >= 9 else digits


async def _find_client_by_phone(phone: str) -> Optional[MSClient]:
    term = _norm_phone(phone)
    if not term or len(term) < 9:
        return None

    async with AsyncSessionSQL() as s:
        stmt = select(MSClient).where(
            or_(
                func.replace(func.replace(MSClient.Telefono, " ", ""), "-", "").like(f"%{term}%"),
                func.replace(func.replace(MSClient.Telefono2, " ", ""), "-", "").like(f"%{term}%"),
                func.replace(func.replace(MSClient.Telefono3, " ", ""), "-", "").like(f"%{term}%"),
            )
        )
        res = await s.execute(stmt)
        return res.scalars().first()


async def _find_client_by_email(email: str) -> Optional[MSClient]:
    e = (email or "").strip().lower()
    if not e or "@" not in e:
        return None
    async with AsyncSessionSQL() as s:
        stmt = select(MSClient).where(or_(func.lower(MSClient.EMail1) == e, func.lower(MSClient.EMail2) == e))
        res = await s.execute(stmt)
        return res.scalars().first()


async def _get_user_by_phone(db: AsyncSession, phone: str) -> Optional[User]:
    p = _norm_phone(phone)
    if not p:
        return None
    res = await db.execute(select(User).where(User.phone.like(f"%{p}%")))
    return res.scalars().first()


async def _get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    e = (email or "").strip().lower()
    if not e:
        return None
    res = await db.execute(select(User).where(func.lower(User.email) == e))
    return res.scalars().first()


def _infer_input_type(filename: str, content_type: str = "") -> InputType:
    fn = (filename or "").lower()
    ct = (content_type or "").lower()

    if ct.startswith("image/") or fn.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return InputType.IMAGE
    if ct.startswith("audio/") or fn.endswith((".mp3", ".wav", ".ogg", ".m4a", ".aac")):
        return InputType.AUDIO
    if fn.endswith(".pdf") or ct == "application/pdf":
        return InputType.PDF
    if fn.endswith((".xlsx", ".xlsm", ".xltx", ".xltm", ".xls")):
        return InputType.XLSX
    if fn.endswith(".docx"):
        return InputType.DOCX
    if fn.endswith((".txt", ".csv", ".md", ".json")):
        return InputType.TEXT
    return InputType.TEXT


def _split_intents(text: str, max_intents: int = 5) -> List[str]:
    """
    Lightweight intent splitting.
    """
    t = (text or "").strip()
    if not t:
        return []
    parts: List[str] = []
    for line in t.splitlines():
        line = line.strip()
        if not line:
            continue
        parts.extend(re.split(r"[.;!?]+", line))

    cleaned: List[str] = []
    seen = set()
    for p in parts:
        p = p.strip()
        if len(p) < 4:
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(p)

    return cleaned[:max_intents] if cleaned else [t]


async def _load_history_whatsapp(db: AsyncSession, client_phone: str, user_phone: str, limit: int) -> List[Chat]:
    stmt = (
        select(Chat)
        .where(Chat.chat_id == client_phone, Chat.user == user_phone)
        .order_by(Chat.id.desc())
        .limit(limit)
    )
    rows = (await db.execute(stmt)).scalars().all()
    return list(reversed(rows))


async def _load_history_email(db: AsyncSession, client_email: str, user_email: str, limit: int) -> List[EmailChat]:
    stmt = (
        select(EmailChat)
        .where(EmailChat.chat_id == client_email, EmailChat.user == user_email)
        .order_by(EmailChat.id.desc())
        .limit(limit)
    )
    rows = (await db.execute(stmt)).scalars().all()
    return list(reversed(rows))


async def _persist_bot_reply_whatsapp(
    db: AsyncSession,
    client_phone: str,
    user_phone: str,
    text_msg: str,
) -> None:
    db.add(
        Chat(
            chat_id=client_phone,
            user=user_phone,
            client=client_phone,
            message=text_msg,
            direction="sent",
            input_type=InputType.TEXT,
            is_bot=True,
            timestamp=_now_utc(),
        )
    )
    await db.commit()


async def _persist_bot_reply_email(
    db: AsyncSession,
    client_email: str,
    user_email: str,
    subject: str,
    text_body: str,
) -> None:
    msg_text = f"Subject: {subject}\n\n{text_body}".strip()
    db.add(
        EmailChat(
            chat_id=client_email,
            user=user_email,
            client=client_email,
            message=msg_text,
            direction="sent",
            input_type=InputType.TEXT,
            is_bot=True,
            timestamp=_now_utc(),
        )
    )
    await db.commit()


# --- Helper: Fake User / Client Generators for Simulation ---

def _fake_user(identifier: str) -> User:
    """Wraps an identifier in a transient User object for simulation."""
    return User(
        id=999999,
        name=f"Sim Salesman {identifier}",
        email=identifier if "@" in identifier else f"{identifier}@fake.local",
        phone=identifier,
        role="user",
        enabled=True
    )

def _fake_client(identifier: str) -> MSClient:
    """Wraps an identifier in a transient MSClient object for simulation."""
    return MSClient(
        CodigoCliente="FAKE001",
        CodigoEmpresa=1,
        Nombre=f"Sim Client {identifier}",
        Telefono=identifier,
        EMail1=identifier if "@" in identifier else f"{identifier}@fake.local"
    )

# -----------------------------------------------------------


async def login_user(user: User) -> Dict[str, Any]:
    import os

    email_monitoring = False
    MOCK_EMAIL = (os.getenv("MOCK_EMAIL") or "").strip().lower()
    SMTP_USER = (os.getenv("SMTP_USER") or "").strip().lower()

    login_email = (user.email or "").strip().lower()
    if not login_email:
        return {"success": False, "error": "user_email_missing"}

    monitored_imap_login = login_email
    logical_user_mailbox = login_email

    if MOCK_EMAIL and SMTP_USER and login_email == MOCK_EMAIL:
        monitored_imap_login = SMTP_USER
        logical_user_mailbox = SMTP_USER

    mailbox_pwd = email_transport.get_app_password(monitored_imap_login)
    if mailbox_pwd:
        email_monitoring = email_transport.start_mailbox(
            mailbox_email=monitored_imap_login,
            mailbox_password=mailbox_pwd,
            user_mailbox=logical_user_mailbox,
        )
    else:
        logger.info(f"No Gmail app password for imap_login={monitored_imap_login}; continuing WhatsApp-only.")

    # 1. Request the Pairing Code from the Go Server
    login_resp = whatsapp_transport.start_login(phone_number=(user.phone or "").strip())

    ok = bool(login_resp.get("success"))
    code = login_resp.get("code") or ""

    if not ok:
        if email_monitoring:
            email_transport.stop_mailbox(logical_user_mailbox)
        return {"success": False, "error": login_resp.get("error") or "start_login_failed"}

    # --- EMAIL SENDING BLOCK ---
    if login_email and code:
        email_transport.send_pairing_code_email(
            to_email=login_email,
            name=user.name or "User",
            code=code
        )

    # 2. Safely Mark the User as Enabled in the Database
    async with AsyncSessionPG() as db:
        
        # Update the specific user who requested the login
        db_user = (await db.execute(select(User).where(User.id == user.id))).scalars().first()
        if db_user:
            db_user.enabled = True
        
        # Maintain secondary SMTP routing if MOCK_EMAIL is used
        smtp_email = logical_user_mailbox.strip().lower()
        if smtp_email != login_email:
            u_mail = (
                await db.execute(select(User).where(func.lower(User.email) == smtp_email))
            ).scalars().first()
            if not u_mail:
                u_mail = User(name="Admin SMTP", email=smtp_email, phone="", role="user", enabled=True)
                db.add(u_mail)
            else:
                u_mail.enabled = True

        await db.commit()

    return {
        "success": True,
        "email_monitoring": email_monitoring,
        "login_user_email": login_email,
        "logical_user_mailbox": logical_user_mailbox,
        "imap_login_email": monitored_imap_login,
        "code": code,
    }


async def logout_user(user: User) -> Dict[str, Any]:
    # Stop email listening
    if user.email:
        email_transport.stop_mailbox(user.email)

    # Actively unlink from WhatsApp servers using the user's phone number
    if user.phone:
        logger.info(f"Sending WhatsApp logout signal for {user.phone}...")
        whatsapp_transport.logout_device(user.phone)
        # Also call delete just in case it was a dead session that needs local cleanup
        whatsapp_transport.delete_device(user.phone)

    async with AsyncSessionPG() as db:
        u = (await db.execute(select(User).where(User.id == user.id))).scalars().first()
        if u:
            u.enabled = False
            u.wa_device_jid = None  # Clear the saved JID
            await db.commit()

    return {"success": True}


async def handle_admin_command(payload: Dict[str, Any]) -> None:
    cmd = (payload.get("command") or "").lower()
    phone = (payload.get("phone") or "").strip()

    if cmd not in ("login", "logout") or not phone:
        return

    now = _now_utc()
    cache_key = f"{phone}_{cmd}"
    last_exec = _admin_command_cache.get(cache_key)
    
    # Only process the command if we haven't processed the exact same one for this user in the last 10 seconds
    if last_exec and (now - last_exec).total_seconds() < 10:
        logger.info(f"Skipping duplicate {cmd} command for {phone} (debounced)")
        return
        
    _admin_command_cache[cache_key] = now
    # ------------------------------

    async with AsyncSessionPG() as db:
        user = await _get_user_by_phone(db, phone)
        if not user:
            logger.warning(f"Admin command target user not found for phone: {phone}")
            whatsapp_transport.send_message(
                to_phone=phone,
                text="❌ *Error:* Tu número de teléfono no está registrado en el sistema como un usuario válido."
            )
            return

        if cmd == "login":
            out = await login_user(user)
            if out.get("success"):
                code = out.get("code")
                if code:
                    reply_msg = f"✅ *Servicio Activado*\nTu código de enlace es: *{code}*\n\nTienes 1 minuto para introducirlo. Ve a Configuración -> Dispositivos Vinculados -> Vincular con el número de teléfono."
                else:
                    reply_msg = "✅ *Servicio Activado*\nEl monitoreo se ha iniciado correctamente (ya estabas conectado)."
            else:
                err = out.get("error", "Error desconocido")
                reply_msg = f"❌ *Error al activar el servicio*\nDetalle: {err}"
        else:
            out = await logout_user(user)
            if out.get("success"):
                reply_msg = "✅ *Servicio Desactivado*\nSe han cerrado tus sesiones y el bot ya no responderá por ti."
            else:
                err = out.get("error", "Error desconocido")
                reply_msg = f"❌ *Error al desactivar el servicio*\nDetalle: {err}"

        # Send the confirmation message back to the user via WhatsApp
        whatsapp_transport.send_message(to_phone=phone, text=reply_msg)

        logger.info(f"Admin cmd {cmd} for {user.email}: {out}")


async def handle_new_message(payload: Dict[str, Any]) -> None:
    """
    WhatsApp ingestion handler.
    Removes MOCK logic; relies on DB lookup or Simulation flags only.
    """
    _prune_caches()
    msg = payload.get("normalized")
    if not msg:
        return
    
    is_simulation = payload.get("is_simulation", False)
    mock_client_force = payload.get("mock_client_force", False)

    from_phone = getattr(msg, "from_phone", "") or ""
    to_phone = getattr(msg, "to_phone", "") or ""
    text_msg = (getattr(msg, "text", "") or "").strip()
    filename = getattr(msg, "filename", "") or ""
    binary = getattr(msg, "binary", b"") or b""

    logger.debug(f"[MSG_FLOW] WA event: from={from_phone} to={to_phone} text={bool(text_msg)}")

    async with AsyncSessionPG() as db:
        # 1. Identify Internal Users
        user_from = await _get_user_by_phone(db, from_phone)
        user_to = await _get_user_by_phone(db, to_phone)

        # --- SELF-CHAT SIMULATION LOGIC ---
        from_norm = _norm_phone(from_phone)
        to_norm = _norm_phone(to_phone)
        
        # If an internal user messages their own number, treat it as a simulation
        is_self_chat = (from_norm == to_norm) and bool(from_norm)
        
        is_mock_owner = False
        if is_self_chat and user_from:
            is_mock_owner = True

        direction: str
        user_phone: str
        client_phone: str
        device_jid: Optional[str] = None
        internal_user: User = None

        if is_mock_owner:
            # Self-chat: simulate a client talking to the bot
            direction = "received"
            user_phone = to_phone
            client_phone = from_phone
            device_jid = getattr(msg, "to_jid", None)
            internal_user = user_to
            logger.info(f"[MSG_FLOW] WA SIMULATION INCOMING (Self-talk by {user_from.email})")
        elif user_from:
            # Salesman sent message
            direction = "sent"
            user_phone = from_phone
            client_phone = to_phone
            device_jid = getattr(msg, "from_jid", None)
            internal_user = user_from
            logger.info(f"[MSG_FLOW] WA OUTGOING (Internal {user_from.email} -> External {to_phone})")

        # Check Admin Command (If the sender is an admin)
        if internal_user.role == "admin":
            m = ADMIN_CMD_RE.match(text_msg or "")
            if m:
                cmd = m.group(1).lower()
                logger.info(f"[MSG_FLOW] Admin Command: {cmd} from {from_phone}")
                await event_bus.emit("admin_command", {"command": cmd, "phone": from_phone})
                return
        
        # Check if the message is sent TO an admin (for a regular user to login/logout)
        # Bypass this block if we are running a self-chat simulation so the AI can answer!
        if user_to and user_to.role == "admin" and not is_mock_owner:
            m = ADMIN_CMD_RE.match(text_msg or "")
            if m:
                cmd = m.group(1).lower()
                logger.info(f"[MSG_FLOW] Admin Command: {cmd} from {from_phone}")
                await event_bus.emit("admin_command", {"command": cmd, "phone": from_phone})
                return 
            else:
                is_help_message = text_msg and ("ChatLink Admin Help" in text_msg)
                if (user_from or is_simulation) and not is_help_message:
                    now = _now_utc()
                    cache_key = f"{from_phone}_{to_phone}"
                    last_sent = _admin_help_cache.get(cache_key)
                    if not last_sent or (now - last_sent).total_seconds() > 10:
                        _admin_help_cache[cache_key] = now
                        logger.info(f"[MSG_FLOW] Sending Admin Help to {from_phone}")
                        whatsapp_transport.send_message(
                            to_phone=from_phone, 
                            text=ADMIN_HELP_TEXT,
                            from_jid=getattr(msg, "to_jid", None)
                        )
                    else:
                        logger.info(f"[MSG_FLOW] Suppressed duplicate Admin Help for {from_phone} (debounced)")
                return

        if not internal_user.enabled and not is_simulation and not is_mock_owner:
            logger.info(f"[MSG_FLOW] WA IGNORED: User {internal_user.email} is DISABLED.")
            return
            
        # 2. Identify Client (Gatekeeper)
        client = await _find_client_by_phone(client_phone)

        # [Simulation] Handle "Not-Client" vs "Mock Client"
        if not client:
            if (is_simulation and mock_client_force) or is_mock_owner:
                client = _fake_client(client_phone)
                logger.info(f"[MSG_FLOW] WA Simulation: Faked client {client_phone}")
            else:
                logger.info(f"[MSG_FLOW] WA DROPPED (Client Not Found): {client_phone}")
                return
        
        logger.info(f"[MSG_FLOW] WA ACCEPTED: Client={getattr(client, 'Nombre', 'Unknown')}")

        if device_jid and not is_simulation:
            internal_user.wa_device_jid = device_jid
            await db.commit()

        # Parse Media
        inferred = _infer_input_type(filename, "")
        extracted = ""
        if binary:
            try:
                if inferred == InputType.IMAGE:
                    extracted = await extract_text_from_image_bytes_async(binary)
                elif inferred == InputType.AUDIO:
                    extracted = await transcribe_audio_bytes_async(binary, filename or "audio.wav")
                else:
                    extracted = await extract_text_from_document_bytes_async(binary, filename or "file.bin")
            except Exception as e:
                logger.warning(f"Media parsing failed: {e}")

        final_text = text_msg
        if extracted:
            final_text = (final_text + "\n\n" if final_text else "") + f"[EXTRACTED]\n{extracted}"

        # --- DEDUPLICATION & ECHO PREVENTION ---
        if not is_simulation:
            # 1. Ignore if it's an echo of a bot message sent in the last 2 minutes
            cutoff_bot = _now_utc() - timedelta(minutes=2)
            is_bot_echo = (
                await db.execute(
                    select(Chat.id).where(
                        Chat.chat_id == client_phone,
                        Chat.user == user_phone,
                        Chat.message == final_text,
                        Chat.is_bot == True,
                        Chat.timestamp >= cutoff_bot,
                    ).limit(1)
                )
            ).scalars().first()

            if is_bot_echo:
                logger.info(f"[MSG_FLOW] WA IGNORED (Bot Echo): {client_phone}")
                return

            # 2. Ignore rapid duplicates (same message within 10 seconds)
            cutoff_rapid = _now_utc() - timedelta(seconds=10)
            is_rapid_dup = (
                await db.execute(
                    select(Chat.id).where(
                        Chat.chat_id == client_phone,
                        Chat.user == user_phone,
                        Chat.message == final_text,
                        Chat.timestamp >= cutoff_rapid,
                    ).limit(1)
                )
            ).scalars().first()

            if is_rapid_dup:
                logger.info(f"[MSG_FLOW] WA IGNORED (Rapid Duplicate): {client_phone}")
                return

        # Insert the fresh, non-echo message
        db.add(
            Chat(
                chat_id=client_phone,
                user=user_phone,
                client=client_phone,
                message=final_text,
                direction=direction,
                input_type=inferred if (binary and inferred) else InputType.TEXT,
                is_bot=False,
                timestamp=_now_utc(),
            )
        )
        await db.commit()

        # FSM Logic: Pass the simulation flag so the AI logic knows to accept the fake client
        sim_flag = is_simulation or is_mock_owner
        if direction == "received":
            await fsm.on_client_message("whatsapp", client_phone, user_phone, is_simulation=sim_flag)
        else:
            await fsm.on_user_message("whatsapp", client_phone, user_phone)


async def handle_new_email(payload: Dict[str, Any]) -> None:
    user_mailbox = (payload.get("user_mailbox") or "").strip().lower()
    from_email = (payload.get("from") or "").strip().lower()
    to_email = (payload.get("to") or "").strip().lower()
    subject = (payload.get("subject") or "").strip()
    body = (payload.get("body") or "").strip()
    attachments = payload.get("attachments") or []
    direction_hint = (payload.get("direction") or "").strip().lower()
    
    is_simulation = payload.get("is_simulation", False)
    mock_client_force = payload.get("mock_client_force", False)

    if not user_mailbox:
        return
    
    logger.debug(f"[MSG_FLOW] Email: mailbox={user_mailbox} from={from_email}")

    async with AsyncSessionPG() as session:
        user = await _get_user_by_email(session, user_mailbox)
        
        # [Simulation] Fake internal user if missing
        if is_simulation and not user:
            user = _fake_user(user_mailbox)

        if not user:
            logger.info(f"[MSG_FLOW] Email DROPPED: Mailbox {user_mailbox} unknown.")
            return
        if not user.enabled and not is_simulation:
            logger.info(f"[MSG_FLOW] Email DROPPED: User {user_mailbox} disabled.")
            return

        if direction_hint in ("sent", "received"):
            direction = direction_hint
            client_email = to_email if direction == "sent" else from_email
        else:
            if from_email == user_mailbox:
                direction = "sent"
                client_email = to_email
            else:
                direction = "received"
                client_email = from_email

        client_email = (client_email or "").strip().lower()
        
        # 2. Identify Client (Gatekeeper) - NO MOCK ENV VARS
        client = await _find_client_by_email(client_email)
        
        if not client:
            if is_simulation and mock_client_force:
                client = _fake_client(client_email)
                logger.info(f"[MSG_FLOW] Email ALLOWED (Simulation Force): {client_email}")
            else:
                logger.info(f"[MSG_FLOW] Email DROPPED: Client {client_email} NOT FOUND in SQL Server.")
                return
        
        extracted_chunks: List[str] = []
        input_type = InputType.TEXT

        for a in attachments:
            fn = (a.get("filename") or "").strip()
            ct = (a.get("content_type") or "").strip()
            data = a.get("bytes") or b""
            if not data:
                continue

            it = _infer_input_type(fn, ct)
            input_type = it if it != InputType.TEXT else input_type

            try:
                if it == InputType.IMAGE:
                    extracted_chunks.append(await extract_text_from_image_bytes_async(data))
                elif it == InputType.AUDIO:
                    extracted_chunks.append(await transcribe_audio_bytes_async(data, fn or "audio.wav"))
                else:
                    extracted_chunks.append(await extract_text_from_document_bytes_async(data, fn or "file.bin"))
            except Exception as e:
                logger.warning(f"Attachment parse failed ({fn}): {e}")

        extracted_text = "\n\n".join([c for c in extracted_chunks if c])
        msg_text = f"Subject: {subject}\n\n{body}".strip()
        if extracted_text:
            msg_text += "\n\n[EXTRACTED]\n" + extracted_text

        # Dedupe Logic
        dup = None
        if not is_simulation:
            window_minutes = int(os.getenv("EMAIL_DEDUPE_WINDOW_MINUTES", "30"))
            cutoff = _now_utc() - timedelta(minutes=window_minutes)
            dup = (
                await session.execute(
                    select(EmailChat.id).where(
                        EmailChat.user == user_mailbox,
                        EmailChat.client == client_email,
                        EmailChat.direction == direction,
                        EmailChat.message == msg_text,
                        EmailChat.timestamp >= cutoff,
                    ).limit(1)
                )
            ).scalars().first()

        if not dup:
            session.add(
                EmailChat(
                    chat_id=client_email,
                    user=user_mailbox,
                    client=client_email,
                    message=msg_text,
                    direction=direction,
                    input_type=input_type,
                    is_bot=False,
                    timestamp=_now_utc(),
                )
            )
            await session.commit()
            logger.info(f"[MSG_FLOW] Email SAVED.")
        else:
            logger.info(f"[MSG_FLOW] Email IGNORED (Duplicate).")

        if direction == "received":
            await fsm.on_client_message("email", client_email, user_mailbox, is_simulation=is_simulation)
        else:
            await fsm.on_user_message("email", client_email, user_mailbox)

async def _build_rag_candidates(queries: List[str], top_k: int = 3) -> Dict[str, Any]:
    """
    Realiza la búsqueda RAG para una lista de queries específicas.
    """
    out: Dict[str, Any] = {}
    if not queries:
        return out

    for q in queries:
        # Validar que q sea un string válido
        if not isinstance(q, str) or not q.strip():
            continue
            
        hits = await rag_service.retrieve(q, top_k=top_k)
        simplified = []
        for h in hits:
            # Normalizar campos para el prompt
            code = h.get("CodigoArticulo") or h.get("id")
            name = h.get("DescripcionArticulo") or h.get("content")
            brand = h.get("MarcaProducto") or ""
            
            simplified.append(
                {
                    "CodigoArticulo": code,
                    "DescripcionArticulo": name,
                    "MarcaProducto": brand,
                    "relevance_score": h.get("relevance_score", 0.0),
                }
            )
        out[q] = simplified
    return out


def _xlsx_from_confirmed_items(confirmed_items: List[Dict[str, Any]]) -> bytes:
    import io
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "order"
    ws.append(["code", "qty"])
    for item in confirmed_items:
        ws.append([item.get("code") or item.get("CodigoArticulo") or "", int(item.get("qty") or 1)])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


async def handle_ai_trigger(payload: Dict[str, Any]) -> None:
    """
    End-to-end Intelligence Orchestrator
    """
    channel = payload.get("channel")
    client_id = payload.get("client_id")
    user_id = payload.get("user_id")
    fired_at = payload.get("fired_at")
    is_simulation = payload.get("is_simulation", False)

    if channel not in ("whatsapp", "email") or not client_id or not user_id:
        return

    logger.info(f"[AI_FLOW] Triggered for {channel} | client={client_id}")
    key = (channel, str(client_id), str(user_id))
    lock = _processing_locks.setdefault(key, asyncio.Lock())

    async with lock:
        try:
            async with AsyncSessionPG() as db:
                # Recuperar usuario interno
                if channel == "whatsapp":
                    user_obj = await _get_user_by_phone(db, user_id)
                else:
                    user_obj = await _get_user_by_email(db, user_id)
                
                if not user_obj and is_simulation:
                    user_obj = _fake_user(user_id)

                salesman_name = getattr(user_obj, "name", "") or "Comercial"

                # --- 1. HISTORY LOAD (FIXED ORDER) ---
                if channel == "whatsapp":
                    if not is_simulation:
                        # Human Takeover Check logic (omitted for brevity, assume unchanged logic if not shown)
                        pass 
                    history = await _load_history_whatsapp(db, client_id, user_id, MAX_HISTORY)
                else:
                    history = await _load_history_email(db, client_id, user_id, MAX_HISTORY)

                # Identify Last User Message explicitly for the Prompt
                # History is now guaranteed [Old, ..., New]
                last_user_text = ""
                # Find the last message that is from the CLIENT (not bot, not sales)
                # Actually, strictly the trigger comes from a received message, so history[-1] should be it.
                if history:
                    last_msg_obj = history[-1]
                    if last_msg_obj.direction == "received" and not last_msg_obj.is_bot:
                        last_user_text = (last_msg_obj.message or "").strip()
                    else:
                        # Fallback search backwards
                        for h in reversed(history):
                            if h.direction == "received" and not h.is_bot:
                                last_user_text = (h.message or "").strip()
                                break
                
                logger.info(f"[AI_FLOW] History size: {len(history)}. Last user input: '{last_user_text}'")

                # --- 2. SUMMARIZATION ---
                last_id = max([h.id for h in history], default=0)
                summary_state = get_summary_state(client_id)
                last_seen_id = int(summary_state.get("_last_seen_id", 0))

                current_summary = summary_state.get("summary") or {}
                previous_status = current_summary.get("order_status", "IDLE")

                new_msgs = [h.message or "" for h in history if h.id > last_seen_id]
                
                updated_summary = await summarize_update_async(current_summary, new_msgs) if new_msgs else current_summary
                set_summary_state(client_id, updated_summary, _last_seen_id=last_id)

                intent = updated_summary.get("last_interaction_intent", "GREETING")
                search_queries = updated_summary.get("search_queries", [])

                # --- SESSION LOGIC ---
                has_bot_spoken_recently = any(h.is_bot for h in history)
                is_new_session = False
                if previous_status in ("IDLE", "CLOSED") and not has_bot_spoken_recently:
                    is_new_session = True
                
                # Exception: Greeting resets intro only if bot hasn't spoken
                if intent == "GREETING" and has_bot_spoken_recently:
                    is_new_session = False

                # Build Context Strings
                recent_chats = history[-10:] if len(history) > 10 else history
                recent_history_lines = []
                for h in recent_chats:
                    sender = "Asistente" if h.is_bot else ("Comercial" if h.direction == "sent" else "Cliente")
                    msg_content = (h.message or "").replace("\n", " ").strip()
                    recent_history_lines.append(f"{sender}: {msg_content}")
                
                recent_history_text = "\n".join(recent_history_lines)

                reply = ""
                
                # --- 3. DECISION LOGIC ---
                
                # 1. Temas ajenos al trabajo (OFF_TOPIC): Silencio absoluto
                if intent == "OFF_TOPIC":
                    logger.info(f"[AI_FLOW] Intent is OFF_TOPIC. Silencing reply for {client_id}.")
                    return
                
                # 2. Peticiones directas al comercial (HUMAN_REQUEST): Mensaje automático y fin
                if intent == "HUMAN_REQUEST":
                    logger.info(f"[AI_FLOW] Intent HUMAN_REQUEST. Sending fallback for {client_id}.")
                    reply = f"{salesman_name} revisará tu mensaje y se pondrá en contacto contigo a la mayor brevedad."
                    
                    # Forzamos el envío directo y cortamos la ejecución para ahorrar llamada al LLM
                    if channel == "whatsapp":
                        from_jid = getattr(user_obj, "wa_device_jid", None) if user_obj else None
                        whatsapp_transport.send_message(to_phone=str(client_id), text=reply, from_jid=from_jid)
                        await _persist_bot_reply_whatsapp(db, str(client_id), str(user_id), reply)
                    else:
                        email_transport.send_email_as(
                            from_email=str(user_id), to_email=str(client_id), subject="Re: Contacto", body=reply
                        )
                        await _persist_bot_reply_email(db, str(client_id), str(user_id), "Re: Contacto", reply)
                    
                    await fsm.on_ai_done(channel, str(client_id), str(user_id))
                    return
                
                # 3. GREETING, ORDER_INTENT y CLOSURE continúan hacia el LLM normalmente
                rag_candidates = {}
                
                if intent != "CLOSURE" and search_queries and isinstance(search_queries, list) and len(search_queries) > 0:
                    logger.info(f"[AI_FLOW] Searching for: {search_queries}")
                    rag_candidates = await _build_rag_candidates(search_queries, top_k=3)
                
                # Recuperar nombre del cliente para naturalidad
                if channel == "whatsapp":
                    client_obj = await _find_client_by_phone(client_id)
                else:
                    client_obj = await _find_client_by_email(client_id)
                
                if not client_obj and is_simulation:
                    client_obj = _fake_client(client_id)
                
                client_name = (getattr(client_obj, "Nombre", "") or "").strip() or "Cliente"

                # Llamada estándar para pedidos en construcción, saludos, human request o cierres
                reply = await build_order_reply_async(
                    client_name=client_name,
                    salesman_name=salesman_name,
                    summary=updated_summary,
                    rag_candidates=rag_candidates,
                    recent_history=recent_history_text,
                    current_message=last_user_text,
                    is_new_session=is_new_session,
                )

                if not reply or not reply.strip():
                    logger.info(f"[AI_FLOW] No reply generated (Silence requested) for {client_id}")
                    return
                
                # --- 4. SEND REPLY TO CLIENT ---
                if channel == "whatsapp":
                    from_jid = getattr(user_obj, "wa_device_jid", None) if user_obj else None
                    
                    send_res = whatsapp_transport.send_message(to_phone=str(client_id), text=reply, from_jid=from_jid)
                    
                    if send_res.get("success") or is_simulation:
                        await _persist_bot_reply_whatsapp(db, str(client_id), str(user_id), reply)
                        logger.info(f"[AI_FLOW] Reply sent/persisted to {client_id}")
                    else:
                        logger.warning(f"WA send failed: {send_res.get('error')}")

                else: # Email Channel
                    ok, err = email_transport.send_email_as(
                        from_email=str(user_id),
                        to_email=str(client_id),
                        subject="Re: Pedido",
                        body=reply,
                    )
                    
                    if ok or is_simulation:
                        await _persist_bot_reply_email(db, str(client_id), str(user_id), "Re: Pedido", reply)
                        logger.info(f"[AI_FLOW] Email Reply sent/persisted to {client_id}")
                
                # --- 5. NOTIFY SALESMAN (Independiente del canal) ---
                target_email = getattr(user_obj, "email", None) if user_obj else None
                
                if target_email and updated_summary.get("order_status") == "CLOSED":
                    items = updated_summary.get("confirmed_items") or []
                    if items:
                        logger.info(f"[AI_FLOW] Pedido cerrado. Generando Excel y enviando a comercial: {target_email}")
                        xlsx = _xlsx_from_confirmed_items(items)
                        
                        # Ejecutamos en un hilo para no bloquear el Event Loop de FastAPI
                        await asyncio.to_thread(
                            email_transport.send_email,
                            to_email=target_email,
                            subject=f"Pedido confirmado ({client_id})",
                            body=f"Pedido confirmado para {client_id}.\n\nItems: {items}",
                            attachments=[("order.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", xlsx)],
                        )

        finally:
            await fsm.on_ai_done(channel, str(client_id), str(user_id))