# chatlink_bot/src/chatlink_bot/handlers.py
"""
Ingestion + AI orchestration.

Changes vs the previous version:
- Bot-echo race fixed: outbound bot texts are fingerprinted in `bot_outbox`
  BEFORE sending; inbound "sent" messages matching a fingerprint are dropped
  before they can be mistaken for a manual salesman reply (which used to
  trigger a handoff — the bot cancelled itself).
- Conversation state moved from ai.llm's in-memory dict to the persisted
  `ConversationSession` row, keyed (channel, client, salesman). Client
  opt-out (`bot_enabled`) and self-introduction (`bot_introduced_at` +
  inactivity gap) are now deterministic columns, not LLM guesses.
- `handle_ai_trigger` calls the single agent loop in ai.llm (search -> ground
  -> reply in one chain) instead of the old Pass A / Pass B pipeline. The
  local `_processing_locks` dict is gone: the FSM is the only concurrency
  authority and never emits two concurrent triggers for the same key.
- Persist-then-send for bot replies (a failed commit no longer leaves an
  unpersisted-but-delivered message).
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from .ai.llm import AgentResult, run_agent
from .ai.parsers import (
    extract_text_from_document_bytes_async,
    extract_text_from_image_bytes_async,
    transcribe_audio_bytes_async,
)
from .ai.rag import rag_service
from .database import AsyncSessionPG, AsyncSessionSQL
from .events import event_bus
from .logic.fsm import fsm
from .logic.lifecycle import conversation_mode, lifecycle_after_turn
from .models import Chat, ConversationSession, EmailChat, InputType, MSClient, User
from .transport.email import email_transport
from .transport.outbox import bot_outbox, same_text, sim_capture
from .logic.sim_clients import sim_client_cache
from .transport.whatsapp import whatsapp_transport

logger = logging.getLogger("Handlers")

MAX_HISTORY = int(os.getenv("BOT_HISTORY_LIMIT", "60"))
SESSION_GAP_HOURS = float(os.getenv("BOT_SESSION_GAP_HOURS", "6"))
HANDOFF_TEXT = os.getenv("BOT_GUARDRAIL_TEXT", "{salesman} revisará tu mensaje y se pondrá en contacto contigo a la mayor brevedad.")
OPTOUT_TEXT = os.getenv("BOT_OPTOUT_TEXT", "Entendido, no te escribiré más 🙂 {salesman} te atenderá personalmente.")

ADMIN_CMD_RE = re.compile(r"^\s*(login|logout)\b", re.IGNORECASE)
ADMIN_HELP_TEXT = (
    "🤖 *ChatLink Admin Help*\n\n"
    "Para gestionar tu acceso, utiliza los siguientes comandos:\n\n"
    "• *login*: Activa el monitoreo de WhatsApp y Email para tu cuenta.\n"
    "• *logout*: Desactiva el servicio y cierra las sesiones activas.\n"
)

_admin_help_cache: Dict[str, datetime] = {}
_admin_command_cache: Dict[str, datetime] = {}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _prune_caches() -> None:
    for cache in (_admin_help_cache, _admin_command_cache):
        if len(cache) > 1000:
            cache.clear()


# --------------------------------------------------------------- identity
def _norm_phone(phone: str) -> str:
    """Loose matching: last 9 digits."""
    digits = re.sub(r"\D", "", phone or "")
    return digits[-9:] if len(digits) >= 9 else digits


async def _find_client_by_phone(phone: str) -> Optional[Any]:
    # 1) Simulation client cache (test entries, never in SAGE). Empty in
    #    production -> one dict lookup, zero behavior change.
    sim = sim_client_cache.resolve(phone)
    if sim is not None:
        logger.info(f"[MSG_FLOW] Client {phone} resolved from SIMULATION cache.")
        return sim
    # 2) SAGE (source of truth).
    term = _norm_phone(phone)
    if len(term) < 9:
        return None
    async with AsyncSessionSQL() as s:
        cols = (MSClient.Telefono, MSClient.Telefono2, MSClient.Telefono3)
        stmt = select(MSClient).where(or_(*(
            func.replace(func.replace(c, " ", ""), "-", "").like(f"%{term}%") for c in cols
        )))
        return (await s.execute(stmt)).scalars().first()


async def _find_client_by_email(email: str) -> Optional[Any]:
    sim = sim_client_cache.resolve(email)      # 1) simulation cache
    if sim is not None:
        logger.info(f"[MSG_FLOW] Client {email} resolved from SIMULATION cache.")
        return sim
    addr = (email or "").strip().lower()       # 2) SAGE
    if "@" not in addr:
        return None
    async with AsyncSessionSQL() as s:
        stmt = select(MSClient).where(
            or_(func.lower(MSClient.EMail1) == addr, func.lower(MSClient.EMail2) == addr)
        )
        return (await s.execute(stmt)).scalars().first()


async def _get_user_by_phone(db: AsyncSession, phone: str) -> Optional[User]:
    p = _norm_phone(phone)
    if not p:
        return None
    return (await db.execute(select(User).where(User.phone.like(f"%{p}%")))).scalars().first()


async def _get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    addr = (email or "").strip().lower()
    if not addr:
        return None
    return (await db.execute(select(User).where(func.lower(User.email) == addr))).scalars().first()


def _fake_user(identifier: str) -> User:
    """Transient salesman object for simulation-only conversations."""
    return User(id=999999, name=f"Sim Salesman {identifier}", phone=identifier, role="user", enabled=True,
                email=identifier if "@" in identifier else f"{identifier}@fake.local")


def _fake_client(identifier: str) -> MSClient:
    """Transient SAGE client object for simulation-only conversations."""
    return MSClient(CodigoCliente="FAKE001", CodigoEmpresa=1, Nombre=f"Sim Client {identifier}",
                    Telefono=identifier, EMail1=identifier if "@" in identifier else f"{identifier}@fake.local")


# ---------------------------------------------------------------- sessions
async def _get_session(db: AsyncSession, channel: str, client_id: str, user_id: str) -> ConversationSession:
    stmt = select(ConversationSession).where(
        ConversationSession.channel == channel,
        ConversationSession.client_id == client_id,
        ConversationSession.user_id == user_id,
    )
    session = (await db.execute(stmt)).scalars().first()
    if session is None:
        session = ConversationSession(channel=channel, client_id=client_id, user_id=user_id)
        db.add(session)
        await db.flush()
    return session


def _gap_expired(session: ConversationSession) -> bool:
    """Client silent longer than SESSION_GAP_HOURS -> the conversation ended by silence."""
    last = session.last_client_msg_at
    return last is None or (_now_utc() - last) > timedelta(hours=SESSION_GAP_HOURS)


# ------------------------------------------------------------ media parsing
def _infer_input_type(filename: str, content_type: str = "", data: bytes = b"") -> InputType:
    fn, ct = (filename or "").lower(), (content_type or "").lower()
    if ct.startswith("image/") or fn.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return InputType.IMAGE
    if ct.startswith("audio/") or fn.endswith((".mp3", ".wav", ".ogg", ".m4a", ".aac")):
        return InputType.AUDIO
    if ct == "application/pdf" or fn.endswith(".pdf"):
        return InputType.PDF
    if fn.endswith((".xlsx", ".xlsm", ".xltx", ".xltm", ".xls")):
        return InputType.XLSX
    if fn.endswith(".docx"):
        return InputType.DOCX
    if fn.endswith((".txt", ".csv", ".md", ".json")):
        return InputType.TEXT
    if data:  # magic bytes for WhatsApp media without extension
        if data.startswith((b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n")) or (data[:4] == b"RIFF" and data[8:12] == b"WEBP"):
            return InputType.IMAGE
        if data.startswith((b"OggS", b"ID3", b"\xff\xfb")):
            return InputType.AUDIO
        if data.startswith(b"%PDF"):
            return InputType.PDF
    return InputType.TEXT


async def _extract_media_text(data: bytes, filename: str, kind: InputType) -> str:
    # Failed recognition must SURFACE, never vanish: the agent gets an explicit
    # marker and (per its prompt) acknowledges it briefly — once, without
    # over-apologizing — and offers an alternative (type it / another format).
    if kind == InputType.IMAGE:
        text = await extract_text_from_image_bytes_async(data)
        return text or "(imagen recibida pero no legible: pide el pedido escrito u otra foto más clara)"
    if kind == InputType.AUDIO:
        text = await transcribe_audio_bytes_async(data, filename or "audio.wav")
        return text or "(audio recibido pero no entendible: pide que lo escriba o lo envíe de otra forma)"
    text = await extract_text_from_document_bytes_async(data, filename or "file.bin")
    return text or "(documento recibido pero no procesable: pide el contenido escrito u otro formato)"


_MEDIA_PREFIX = {InputType.IMAGE: "[Texto en Imagen]:", InputType.AUDIO: "[Audio transcrito]:"}


# ---------------------------------------------------------------- persistence
async def _load_history(db: AsyncSession, channel: str, client_id: str, user_id: str, limit: int) -> List[Any]:
    model = Chat if channel == "whatsapp" else EmailChat
    stmt = (select(model).where(model.chat_id == client_id, model.user == user_id)
            .order_by(model.id.desc()).limit(limit))
    return list(reversed((await db.execute(stmt)).scalars().all()))


async def _persist_bot_reply(db: AsyncSession, channel: str, client_id: str, user_id: str, text_msg: str) -> None:
    model = Chat if channel == "whatsapp" else EmailChat
    db.add(model(chat_id=client_id, user=user_id, client=client_id, message=text_msg,
                 direction="sent", input_type=InputType.TEXT, is_bot=True, timestamp=_now_utc()))
    await db.commit()


def _history_lines(history: List[Any], last: int = 10) -> str:
    lines = []
    for h in history[-last:]:
        sender = "Asistente" if h.is_bot else ("Comercial" if h.direction == "sent" else "Cliente")
        lines.append(f"{sender}: {(h.message or '').replace(chr(10), ' ').strip()}")
    return "\n".join(lines)


# --------------------------------------------------------------- admin flows
async def login_user(user: User) -> Dict[str, Any]:
    """Pair the salesman's WhatsApp device and start their mailbox listener."""
    email_monitoring = False
    mock_email = (os.getenv("MOCK_EMAIL") or "").strip().lower()
    smtp_user = (os.getenv("SMTP_USER") or "").strip().lower()

    login_email = (user.email or "").strip().lower()
    if not login_email:
        return {"success": False, "error": "user_email_missing"}

    # MOCK_EMAIL lets a dev route a user's logical mailbox through the admin Gmail.
    imap_login = smtp_user if (mock_email and smtp_user and login_email == mock_email) else login_email
    user_mailbox = imap_login

    mailbox_pwd = email_transport.get_app_password(imap_login)
    if mailbox_pwd:
        email_monitoring = email_transport.start_mailbox(
            mailbox_email=imap_login, mailbox_password=mailbox_pwd, user_mailbox=user_mailbox)
    else:
        logger.info(f"No Gmail app password for {imap_login}; continuing WhatsApp-only.")

    login_resp = whatsapp_transport.start_login(phone_number=(user.phone or "").strip())
    code = login_resp.get("code") or ""
    if not login_resp.get("success"):
        if email_monitoring:
            email_transport.stop_mailbox(user_mailbox)
        return {"success": False, "error": login_resp.get("error") or "start_login_failed"}

    if code:
        email_transport.send_pairing_code_email(to_email=login_email, name=user.name or "User", code=code)

    async with AsyncSessionPG() as db:
        db_user = (await db.execute(select(User).where(User.id == user.id))).scalars().first()
        if db_user:
            db_user.enabled = True
        if user_mailbox != login_email:  # keep secondary SMTP routing user enabled
            u_mail = (await db.execute(select(User).where(func.lower(User.email) == user_mailbox))).scalars().first()
            if not u_mail:
                db.add(User(name="Admin SMTP", email=user_mailbox, phone="", role="user", enabled=True))
            else:
                u_mail.enabled = True
        await db.commit()

    return {"success": True, "email_monitoring": email_monitoring, "code": code,
            "login_user_email": login_email, "logical_user_mailbox": user_mailbox, "imap_login_email": imap_login}


async def logout_user(user: User) -> Dict[str, Any]:
    if user.email:
        email_transport.stop_mailbox(user.email)
    if user.phone:
        logger.info(f"Sending WhatsApp logout for {user.phone}...")
        whatsapp_transport.logout_device(user.phone)
        whatsapp_transport.delete_device(user.phone)  # cleanup dead sessions too
    async with AsyncSessionPG() as db:
        db_user = (await db.execute(select(User).where(User.id == user.id))).scalars().first()
        if db_user:
            db_user.enabled = False
            db_user.wa_device_jid = None
            await db.commit()
    return {"success": True}


async def handle_admin_command(payload: Dict[str, Any]) -> None:
    cmd = (payload.get("command") or "").lower()
    phone = (payload.get("phone") or "").strip()
    reply_from_jid = payload.get("reply_from_jid") or None
    if cmd not in ("login", "logout") or not phone:
        return

    now, cache_key = _now_utc(), f"{phone}_{cmd}"
    last = _admin_command_cache.get(cache_key)
    if last and (now - last).total_seconds() < 10:
        logger.info(f"Skipping duplicate {cmd} for {phone} (debounced).")
        return
    _admin_command_cache[cache_key] = now

    async with AsyncSessionPG() as db:
        user = await _get_user_by_phone(db, phone)
    if not user:
        whatsapp_transport.send_message(
            to_phone=phone, text="❌ *Error:* Tu número no está registrado como usuario válido.",
            from_jid=reply_from_jid)
        return

    if cmd == "login":
        out = await login_user(user)
        if out.get("success"):
            code = out.get("code")
            reply = (f"✅ *Servicio Activado*\nTu código de enlace es: *{code}*\n\nTienes 1 minuto. "
                     "Ve a Configuración -> Dispositivos Vinculados -> Vincular con el número de teléfono."
                     if code else "✅ *Servicio Activado*\nEl monitoreo se ha iniciado (ya estabas conectado).")
        else:
            reply = f"❌ *Error al activar el servicio*\nDetalle: {out.get('error', 'desconocido')}"
    else:
        out = await logout_user(user)
        reply = ("✅ *Servicio Desactivado*\nSesiones cerradas; el bot ya no responderá por ti."
                 if out.get("success") else f"❌ *Error al desactivar*\nDetalle: {out.get('error', 'desconocido')}")

    whatsapp_transport.send_message(to_phone=phone, text=reply, from_jid=reply_from_jid)
    logger.info(f"Admin cmd {cmd} for {user.email} (reply via {reply_from_jid or 'default'}): {out}")


# ------------------------------------------------------------ WhatsApp ingest
async def handle_new_message(payload: Dict[str, Any]) -> None:
    _prune_caches()
    msg = payload.get("normalized")
    if not msg:
        return
    is_simulation: bool = payload.get("is_simulation", False)
    mock_client_force: bool = payload.get("mock_client_force", False)

    from_phone: str = getattr(msg, "from_phone", "") or ""
    to_phone: str = getattr(msg, "to_phone", "") or ""
    text_msg: str = (getattr(msg, "text", "") or "").strip()
    filename: str = getattr(msg, "filename", "") or ""
    binary: bytes = getattr(msg, "binary", b"") or b""

    async with AsyncSessionPG() as db:
        user_from = await _get_user_by_phone(db, from_phone)
        user_to = await _get_user_by_phone(db, to_phone)

        # Self-chat = a linked user messaging their own number -> full-workflow simulation.
        is_self_chat = bool(_norm_phone(from_phone)) and _norm_phone(from_phone) == _norm_phone(to_phone)
        is_mock_owner = is_self_chat and user_from is not None

        # DIAGNOSTIC: shows exactly why a message is (or isn't) classified as
        # self-talk and how the admin router will see it — from/to numbers,
        # their resolved users+roles, and the flags. When "salesman -> admin"
        # is being treated as self-talk, this line reveals whether from==to
        # (same physical number) or a role/lookup surprise is the cause.
        logger.info(
            "[MSG_FLOW][DIAG] from=%s (user=%s role=%s) -> to=%s (user=%s role=%s) | "
            "self_chat=%s mock_owner=%s text=%r",
            from_phone, getattr(user_from, "email", None), getattr(user_from, "role", None),
            to_phone, getattr(user_to, "email", None), getattr(user_to, "role", None),
            is_self_chat, is_mock_owner, (text_msg or "")[:40])

        if is_mock_owner:
            direction, user_phone, client_phone = "received", to_phone, from_phone
            device_jid, internal_user = getattr(msg, "to_jid", None), (user_to or user_from)
            logger.info(f"[MSG_FLOW] WA SIMULATION INCOMING (self-talk by {user_from.email})")
        elif user_from:  # salesman wrote (manually, or it is the bot's own echo)
            direction, user_phone, client_phone = "sent", from_phone, to_phone
            device_jid, internal_user = getattr(msg, "from_jid", None), user_from
        elif user_to:  # normal inbound: external client -> salesman
            direction, user_phone, client_phone = "received", to_phone, from_phone
            device_jid, internal_user = getattr(msg, "to_jid", None), user_to
        else:
            logger.debug(f"[MSG_FLOW] WA IGNORED (no internal user): {from_phone} -> {to_phone}")
            return

        # ECHO GUARD. Production (normal chats): the bot sends from the
        # salesman's JID, so its messages echo back as direction="sent" —
        # unchanged, sent-only semantics. SELF-CHAT (full mode) additionally
        # checks all directions, because there the echo is classified as a
        # CLIENT message ("received") and would otherwise start an infinite
        # bot-answers-itself loop. Fingerprints are registered BEFORE sending.
        if (direction == "sent" or is_self_chat) and \
                bot_outbox.is_echo(("whatsapp", client_phone, user_phone), text_msg):
            logger.info(f"[MSG_FLOW] WA IGNORED (bot echo) for {client_phone}.")
            return

        # Admin command channel. login/logout are ALWAYS admin operations for
        # the SENDER's own account — never client orders — so they must reach
        # the admin handler and NEVER the AI. This includes SELF-TALK: the log
        # showed a "login" self-talk falling through to Kapa, which then replied
        # salesman->salesman. The reply is sent from the admin/receiving device
        # (in self-talk that is the user's own device, but the content is the
        # login code, not a Kapa answer).
        # Admin channel. login/logout always go to the admin handler, never the
        # AI. THE DEVICE MATTERS MORE THAN THE ADDRESS: msg.to_jid on a message
        # observed on the SENDER's device is just the recipient address — NOT a
        # device the meow server can send from. Passing it made the server fall
        # back silently to the default linked device (the salesman's own phone),
        # which is why every reply kept arriving salesman -> salesman no matter
        # how the routing was fixed. The only JID the server can actually send
        # from is a LINKED device, and the admin's linked device is stored on
        # their User row (wa_device_jid, captured at login). Use THAT.
        admin_cmd = ADMIN_CMD_RE.match(text_msg)
        sender_is_admin = bool(internal_user and internal_user.role == "admin")
        to_is_admin = bool(user_to and user_to.role == "admin")
        admin_user = user_to if to_is_admin else (internal_user if sender_is_admin else None)
        admin_device = getattr(admin_user, "wa_device_jid", None) if admin_user else None
        if admin_cmd and (is_mock_owner or sender_is_admin or to_is_admin):
            if (is_mock_owner or sender_is_admin) and not to_is_admin:
                # command issued on the user's own device (self-talk / admin self)
                admin_device = admin_device or getattr(msg, "from_jid", None)
            await event_bus.emit("admin_command", {
                "command": admin_cmd.group(1).lower(), "phone": from_phone,
                "reply_from_jid": admin_device})
            dev_desc = admin_device or "DEFAULT (admin has no linked device!)"
            logger.info(f"[MSG_FLOW] Admin command '{admin_cmd.group(1).lower()}' routed: "
                        f"reply FROM device {dev_desc} -> TO {from_phone}.")
            return
        # A plain (non-command) message to an admin number -> help card,
        # also from the admin's LINKED device.
        if to_is_admin and not is_mock_owner:
            if (user_from or is_simulation) and "ChatLink Admin Help" not in text_msg:
                now, cache_key = _now_utc(), f"{from_phone}_{to_phone}"
                last = _admin_help_cache.get(cache_key)
                if not last or (now - last).total_seconds() > 10:
                    _admin_help_cache[cache_key] = now
                    if not admin_device:
                        logger.warning(f"[MSG_FLOW] Admin {getattr(user_to, 'email', to_phone)} has "
                                       "no linked WhatsApp device (wa_device_jid empty): the help "
                                       "reply will go out from the DEFAULT device and look "
                                       "self-sent. Link the admin number (send 'login' from it).")
                    whatsapp_transport.send_message(to_phone=from_phone, text=ADMIN_HELP_TEXT,
                                                    from_jid=admin_device)
            return

        if not internal_user.enabled and not is_simulation and not is_mock_owner:
            logger.info(f"[MSG_FLOW] WA IGNORED: user {internal_user.email} is DISABLED.")
            return

        # Gatekeeper: only SAGE clients ever get stored or answered.
        client = await _find_client_by_phone(client_phone)
        if not client:
            if (is_simulation and mock_client_force) or is_mock_owner:
                client = _fake_client(client_phone)
            else:
                logger.info(f"[MSG_FLOW] WA DROPPED (client not in SAGE): {client_phone}")
                return

        if device_jid and not is_simulation:
            internal_user.wa_device_jid = device_jid
            await db.commit()

        # Media -> text.
        inferred = _infer_input_type(filename, "", binary)
        final_text = text_msg
        if binary:
            try:
                extracted = await _extract_media_text(binary, filename, inferred)
                if extracted:
                    prefix = _MEDIA_PREFIX.get(inferred, "[Documento]:")
                    final_text = (final_text + "\n" if final_text else "") + f"{prefix} {extracted}"
            except Exception as e:
                logger.warning(f"Media parsing failed: {e}")

        # Rapid-duplicate guard (transport retries).
        if not is_simulation:
            cutoff = _now_utc() - timedelta(seconds=10)
            dup = (await db.execute(select(Chat.id).where(
                Chat.chat_id == client_phone, Chat.user == user_phone,
                Chat.message == final_text, Chat.timestamp >= cutoff).limit(1))).scalars().first()
            if dup:
                logger.info(f"[MSG_FLOW] WA IGNORED (rapid duplicate): {client_phone}")
                return

        db.add(Chat(chat_id=client_phone, user=user_phone, client=client_phone, message=final_text,
                    direction=direction, input_type=inferred if binary else InputType.TEXT,
                    is_bot=False, timestamp=_now_utc()))
        await db.commit()

    sim_flag = is_simulation or is_mock_owner
    if direction == "received":
        await fsm.on_client_message("whatsapp", client_phone, user_phone, is_simulation=sim_flag)
    else:
        await fsm.on_user_message("whatsapp", client_phone, user_phone)


# --------------------------------------------------------------- Email ingest
async def handle_new_email(payload: Dict[str, Any]) -> None:
    user_mailbox = (payload.get("user_mailbox") or "").strip().lower()
    from_email = (payload.get("from") or "").strip().lower()
    to_email = (payload.get("to") or "").strip().lower()
    subject = (payload.get("subject") or "").strip()
    body = (payload.get("body") or "").strip()
    attachments: List[Dict[str, Any]] = payload.get("attachments") or []
    direction_hint = (payload.get("direction") or "").strip().lower()
    is_simulation: bool = payload.get("is_simulation", False)
    mock_client_force: bool = payload.get("mock_client_force", False)

    if not user_mailbox:
        return

    async with AsyncSessionPG() as db:
        user = await _get_user_by_email(db, user_mailbox)
        if not user and is_simulation:
            user = _fake_user(user_mailbox)
        if not user:
            logger.info(f"[MSG_FLOW] Email DROPPED: mailbox {user_mailbox} unknown.")
            return
        if not user.enabled and not is_simulation:
            logger.info(f"[MSG_FLOW] Email DROPPED: user {user_mailbox} disabled.")
            return

        # FULL MODE (self-chat): the salesman emailing their OWN mailbox acts
        # as the client and the bot answers from the same account — mirrors
        # the WhatsApp self-chat path. Echo guard + dedupe prevent loops.
        is_self_email = bool(user_mailbox) and from_email == user_mailbox and to_email == user_mailbox

        if is_self_email:
            direction = "received"
        elif direction_hint in ("sent", "received"):
            direction = direction_hint
        else:
            direction = "sent" if from_email == user_mailbox else "received"
        client_email = (to_email if direction == "sent" else from_email).strip().lower()

        # ECHO GUARD: production keeps sent-only semantics (sent-folder polls);
        # SELF-CHAT additionally checks all directions — there the bot's own
        # reply comes back classified as a CLIENT message, via inbox AND
        # sent-folder polls.
        chunks_probe = f"Subject: {subject}\n\n{body}".strip()
        if (direction == "sent" or is_self_email) and \
                bot_outbox.is_echo(("email", client_email, user_mailbox), chunks_probe):
            logger.info(f"[MSG_FLOW] Email IGNORED (bot echo) for {client_email}.")
            return

        # Gatekeeper: only SAGE clients (the owner acts as a fake client in full mode).
        client = await _find_client_by_email(client_email)
        if not client:
            if (is_simulation and mock_client_force) or is_self_email:
                client = _fake_client(client_email)
            else:
                logger.info(f"[MSG_FLOW] Email DROPPED: client {client_email} not in SAGE.")
                return

        # Attachments -> text.
        chunks: List[str] = []
        input_type = InputType.TEXT
        for a in attachments:
            data: bytes = a.get("bytes") or b""
            if not data:
                continue
            kind = _infer_input_type(a.get("filename") or "", a.get("content_type") or "", data)
            input_type = kind if kind != InputType.TEXT else input_type
            try:
                chunks.append(await _extract_media_text(data, a.get("filename") or "", kind))
            except Exception as e:
                logger.warning(f"Attachment parse failed ({a.get('filename')}): {e}")

        msg_text = f"Subject: {subject}\n\n{body}".strip()
        extracted = "\n\n".join(c for c in chunks if c)
        if extracted:
            msg_text += "\n\n[EXTRACTED]\n" + extracted

        # Duplicate guard: exact repeats must not re-trigger the FSM either.
        if not is_simulation:
            cutoff = _now_utc() - timedelta(minutes=int(os.getenv("EMAIL_DEDUPE_WINDOW_MINUTES", "30")))
            dup = (await db.execute(select(EmailChat.id).where(
                EmailChat.user == user_mailbox, EmailChat.client == client_email,
                EmailChat.direction == direction, EmailChat.message == msg_text,
                EmailChat.timestamp >= cutoff).limit(1))).scalars().first()
            if dup:
                logger.info("[MSG_FLOW] Email IGNORED (duplicate).")
                return

        db.add(EmailChat(chat_id=client_email, user=user_mailbox, client=client_email, message=msg_text,
                         direction=direction, input_type=input_type, is_bot=False, timestamp=_now_utc()))
        await db.commit()

    if direction == "received":
        await fsm.on_client_message("email", client_email, user_mailbox,
                                    is_simulation=is_simulation or is_self_email)
    else:
        await fsm.on_user_message("email", client_email, user_mailbox)


# -------------------------------------------------------------------- RAG glue
async def _build_rag_candidates(queries: List[str], top_k: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """Run retrieval per query and normalize hits for the agent prompt."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for q in queries:
        if not isinstance(q, str) or not q.strip():
            continue
        hits = await rag_service.retrieve(q, top_k=top_k)
        out[q] = [{
            "CodigoArticulo": h.get("CodigoArticulo") or h.get("id"),
            "DescripcionArticulo": h.get("DescripcionArticulo") or h.get("content"),
            "MarcaProducto": h.get("MarcaProducto") or "",
            "relevance_score": h.get("relevance_score", 0.0),
            "covers_query": bool(h.get("covers_query", False)),
            "query_uncovered": list(h.get("query_uncovered") or []),
        } for h in hits]
    return out


def _xlsx_from_cart(cart: List[Dict[str, Any]]) -> bytes:
    import io
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "order"
    ws.append(["code", "qty"])
    for item in cart:
        ws.append([item.get("code") or "", int(item.get("qty") or 1)])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ------------------------------------------------------------------ AI trigger
def _send_reply(channel: str, client_id: str, user_id: str, reply: str,
                user_obj: Optional[User]) -> Dict[str, Any]:
    """Fingerprint (echo guard) then hand the reply to the real transport.
    Self-conversations (full mode) register extra fingerprint copies: one send
    can echo back through several paths (WhatsApp device sync; email inbox AND
    sent-folder pollers), and each is_echo() hit consumes one copy."""
    is_self = (_norm_phone(client_id) == _norm_phone(user_id)) if channel == "whatsapp" \
        else (client_id.strip().lower() == user_id.strip().lower())
    copies = 3 if is_self else 1
    if channel == "whatsapp":
        bot_outbox.register(("whatsapp", client_id, user_id), reply, copies=copies)
        return whatsapp_transport.send_message(
            to_phone=client_id, text=reply, from_jid=getattr(user_obj, "wa_device_jid", None))
    subject = "Re: Pedido"
    bot_outbox.register(("email", client_id, user_id),
                        f"Subject: {subject}\n\n{reply}".strip(), copies=copies)  # match ingest format
    ok, err = email_transport.send_email_as(from_email=user_id, to_email=client_id, subject=subject, body=reply)
    return {"success": ok, "error": err}


async def handle_ai_trigger(payload: Dict[str, Any]) -> None:
    """Debounce fired: run the agent for one conversation turn and deliver its reply."""
    channel: str = payload.get("channel") or ""
    client_id: str = str(payload.get("client_id") or "")
    user_id: str = str(payload.get("user_id") or "")
    is_simulation: bool = payload.get("is_simulation", False)
    if channel not in ("whatsapp", "email") or not client_id or not user_id:
        return

    logger.info(f"[AI_FLOW] Triggered for {channel} | client={client_id} user={user_id}")
    try:
        async with AsyncSessionPG() as db:
            # 1. Actors + persisted session.
            user_obj = await (_get_user_by_phone(db, user_id) if channel == "whatsapp"
                              else _get_user_by_email(db, user_id))
            if not user_obj and is_simulation:
                user_obj = _fake_user(user_id)
            salesman_name = getattr(user_obj, "name", "") or "Comercial"

            session = await _get_session(db, channel, client_id, user_id)
            if not session.bot_enabled:
                logger.info(f"[AI_FLOW] Bot OPTED-OUT for {client_id}; staying silent.")
                return

            # 2. History + the messages this turn must answer: the trailing run of
            #    client messages since the last bot/salesman message. (An id cursor
            #    would treat a client's entire pre-bot history as "new" on the
            #    bot's first turn for that conversation.)
            history = await _load_history(db, channel, client_id, user_id, MAX_HISTORY)
            last_client_msg = next((h for h in reversed(history)
                                    if h.direction == "received" and not h.is_bot), None)
            pending: List[str] = []
            for h in reversed(history):
                if h.direction == "received" and not h.is_bot:
                    pending.append((h.message or "").strip())
                else:
                    break
            # BATCHING: `pending` is the whole trailing run of client messages
            # since the last bot/salesman turn — every message that arrived
            # during the debounce window is answered together, in ONE turn.
            # If it is empty, there is nothing NEW to answer: a spurious or
            # duplicate trigger (e.g. a re-debounce after a message that the
            # previous turn already folded in). Answering the old last message
            # again is exactly the "responds twice" bug — stay silent instead.
            if not pending:
                logger.info(f"[AI_FLOW] No new client messages since last reply for "
                            f"{channel}/{client_id}; nothing to answer, staying silent.")
                return
            if len(pending) > 1:
                logger.info(f"[AI_FLOW] Batching {len(pending)} client messages into one turn "
                            f"for {channel}/{client_id}.")
            current_message = "\n".join(reversed(pending)).strip()

            # LOOP BREAKER (last resort): if the newest "client" text is
            # byte-identical to the bot's own last reply, an echo leaked past
            # the outbox guard (transport transformed it, TTL expired, or a
            # restart wiped the fingerprints). Answering it would start a
            # self-feeding loop — stay silent instead.
            last_bot_msg = next((h for h in reversed(history) if h.is_bot), None)
            if last_bot_msg and current_message and same_text(current_message, last_bot_msg.message or ""):
                logger.warning(f"[AI_FLOW] Loop guard: client text equals my own last reply "
                               f"({channel}/{client_id}); staying silent.")
                return

            # 3. Conversation lifecycle — decided IN CODE by logic/lifecycle.py;
            #    the LLM only decides the client's intent within its mode. A
            #    dispatched (CLOSED) order is immutable: on any ended
            #    conversation the agent works on a FRESH cart, with the last
            #    dispatched order as context ("ponme lo mismo que la última vez").
            if session.order_status == "CLOSED" and session.cart:
                session.last_closed_cart = session.cart          # stash the dispatched order
            intro_mode, ended = conversation_mode(
                introduced=session.bot_introduced_at is not None,
                conv_open=bool(session.conv_open),
                order_status=session.order_status or "IDLE",
                gap_expired=_gap_expired(session),
            )
            working_status = "IDLE" if ended else (session.order_status or "IDLE")
            working_cart: List[Dict[str, Any]] = [] if ended else (session.cart or [])
            working_summary = "" if ended else (session.summary or "")
            # The multi-item enrichment queue and the one-time capability
            # guide are per-conversation: an ended conversation starts clean.
            # getattr fallbacks: tolerate a ConversationSession row that
            # predates the open_items / guide_shown columns (stale schema).
            working_open_items: List[Dict[str, Any]] = [] if ended else (getattr(session, "open_items", None) or [])
            working_guide_shown = False if ended else bool(getattr(session, "guide_shown", False))

            client_obj = await (_find_client_by_phone(client_id) if channel == "whatsapp"
                                else _find_client_by_email(client_id))
            if not client_obj and is_simulation:
                client_obj = _fake_client(client_id)
            client_name = (getattr(client_obj, "Nombre", "") or "").strip() or "Cliente"

            # 4. Agent turn (search -> ground -> reply in one loop).
            result: AgentResult = await run_agent(
                client_name=client_name,
                salesman_name=salesman_name,
                session={"order_status": working_status, "cart": working_cart,
                         "summary": working_summary,
                         "open_items": working_open_items,
                         "guide_shown": working_guide_shown,
                         "last_closed_cart": session.last_closed_cart or []},
                recent_history=_history_lines(history),
                current_message=current_message,
                intro_mode=intro_mode,
                search=_build_rag_candidates,
            )

            # 5. Decide the outgoing text first (canned messages are code, not
            #    LLM output) — the lifecycle below depends on what is delivered.
            if result.opt_out:
                reply = OPTOUT_TEXT.format(salesman=salesman_name)
            elif result.handoff:
                # handoff + silent = Pass-1 triage ESC_HANDOFF (explicit human
                # demand / anger / uncommercial dispute): strict silence — no
                # canned line, no LLM text — the salesman takes over clean.
                # handoff without silent keeps the previous semantics: PURE
                # escalation decided in the agent loop -> canned referral.
                if result.silent:
                    reply = ""
                else:
                    reply = result.reply.strip() or HANDOFF_TEXT.format(salesman=salesman_name)
            else:
                reply = result.reply.strip()
            spoke_as_kapa = bool(reply) and not (result.opt_out or (result.handoff and not result.reply.strip()))

            # 6. Persist lifecycle BEFORE sending (a failed commit must not
            #    leave a delivered-but-unrecorded reply). Rules live in
            #    logic/lifecycle.py (unit-tested there).
            outcome = lifecycle_after_turn(
                intro_mode=intro_mode, spoke_as_kapa=spoke_as_kapa,
                result_status=result.order_status, cart_size=len(result.cart),
                handoff=result.handoff, opt_out=result.opt_out,
            )
            session.order_status, session.cart, session.summary = result.order_status, result.cart, result.summary
            # These two columns may be absent on an out-of-date table; only
            # assign when the mapper actually has them (a bare setattr on an
            # ORM object without the column would silently never persist).
            if hasattr(type(session), "open_items"):
                session.open_items = result.open_items
            if hasattr(type(session), "guide_shown"):
                session.guide_shown = bool(result.guide_shown)
            if result.order_status == "CLOSED":
                session.last_closed_cart = result.cart
            if result.opt_out:
                session.bot_enabled = False
            if outcome.conv_open is not None:
                session.conv_open = outcome.conv_open
            if outcome.introduced_now:
                session.bot_introduced_at = _now_utc()
            session.last_client_msg_at = getattr(last_client_msg, "timestamp", None) or _now_utc()
            await db.commit()

            if not reply:
                logger.info(f"[AI_FLOW] Silence for {client_id}.")
                return

            await _persist_bot_reply(db, channel, client_id, user_id, reply)
            send_res = _send_reply(channel, client_id, user_id, reply, user_obj)
            if not send_res.get("success") and not is_simulation:
                logger.warning(f"[AI_FLOW] Send failed for {client_id}: {send_res.get('error')}")
            if is_simulation:
                sim_capture.record((channel, client_id, user_id), reply,
                                   meta={"ctx": getattr(result, "ctx", {})})

            # 7. Salesman notifications. A closed order ships as xlsx. Handoffs
            #    do NOT email the salesman (he sees the conversation on his own
            #    WhatsApp). The email goes out for any REAL salesman inbox —
            #    including full-mode self-talk, which exists precisely to verify
            #    end-to-end delivery. Only the fake simulation fixtures
            #    (sales@sim.com / admin@sim.com) have no inbox and are skipped.
            target_email = getattr(user_obj, "email", None)
            is_fake_sim_actor = bool(target_email) and \
                target_email.strip().lower().endswith("@sim.com")
            if result.order_status == "CLOSED" and result.cart:
                if not target_email:
                    logger.error(f"[AI_FLOW] Order CLOSED for {client_id} but salesman has no "
                                 "email on file; cannot deliver the xlsx.")
                elif is_fake_sim_actor:
                    logger.info(f"[AI_FLOW] Order CLOSED; salesman {target_email} is a sim "
                                "fixture with no inbox — xlsx email skipped.")
                else:
                    logger.info(f"[AI_FLOW] Order CLOSED; emailing xlsx to {target_email} with "
                                f"{len(result.cart)} rows: {[i.get('code') for i in result.cart]}")
                    ok, err = await asyncio.to_thread(
                        email_transport.send_email,
                        to_email=target_email,
                        subject=f"Pedido confirmado ({client_id})",
                        body=f"Pedido confirmado para {client_id} ({client_name}).\n\n"
                             f"Items: {result.cart}",
                        attachments=[("order.xlsx",
                                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                      _xlsx_from_cart(result.cart))],
                    )
                    if ok:
                        logger.info(f"[AI_FLOW] Order xlsx delivered to {target_email}.")
                    else:
                        logger.error(f"[AI_FLOW] Order xlsx FAILED to send to {target_email}: {err}")
    except Exception as e:
        logger.exception(f"[AI_FLOW] Unhandled error for {channel}/{client_id}: {e}")
    finally:
        await fsm.on_ai_done(channel, client_id, user_id)