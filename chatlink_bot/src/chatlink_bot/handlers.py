# chatlink_bot/src/chatlink_bot/handlers.py
"""
Ingestion, admin commands and the conversation turn.

Ingestion only STORES messages. logic/loop.py decides when a conversation is
due and calls run_turn(); replies and closed orders are written in the same
transaction as the session and delivered by send_queued()/send_orders().
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from functools import partial
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
from .ai.rag import client_history, normalize_text
from .database import AsyncSessionPG, AsyncSessionSQL
from .events import event_bus
from .logic.alerts import alert
from .logic.lifecycle import conversation_mode, lifecycle_after_turn
from .models import (Chat, ClientAlias, ConversationSession, EmailChat, InputType, KapaOrder,
                     MSArticle, MSClient, User)
from .transport.email import email_transport
from .transport.whatsapp import whatsapp_transport

logger = logging.getLogger("Handlers")

MAX_HISTORY = int(os.getenv("BOT_HISTORY_LIMIT", "60"))
SESSION_GAP_HOURS = float(os.getenv("BOT_SESSION_GAP_HOURS", "6"))
HANDOFF_TEXT = os.getenv("BOT_GUARDRAIL_TEXT", "{salesman} revisará tu mensaje y se pondrá en contacto contigo a la mayor brevedad.")
OPTOUT_TEXT = os.getenv("BOT_OPTOUT_TEXT", "Entendido, no te escribiré más 🙂 {salesman} te atenderá personalmente.")

ADMIN_CMD_RE = re.compile(r"^\s*(login|logout)\b", re.IGNORECASE)
# Direct address to the bot by name re-engages it after an opt-out.
BOT_NAME = os.getenv("AGENT_BOT_NAME", "Kapa").strip()
BOT_NAME_RE = re.compile(rf"\b{re.escape(BOT_NAME)}\b", re.IGNORECASE) if BOT_NAME else None
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


def _only_digits(value: str) -> str:
    return re.sub(r"\D", "", value or "")


# Punctuation that manual data entry puts inside SAGE phone numbers. Stripped on
# BOTH sides (SQL and Python) before comparing.
_PHONE_PUNCT = (" ", "-", ".", "(", ")", "+", "/")
# Cap on candidate rows. If a phone matches more than this, the data is bad
# enough that we do not want to resolve it automatically anyway.
_MAX_CANDIDATE_ROWS = 25


@dataclass
class ClientIdentity:
    """Result of resolving an identifier (phone or email) against SAGE.

    WHY THIS EXISTS. The previous lookup was
        LIKE '%last 9 digits%' over Telefono/Telefono2/Telefono3
        ... .scalars().first()   with no ORDER BY
    and a wrong hit only got the greeting name wrong. The moment product
    resolution leans on that client's ORDER HISTORY, a wrong hit makes the bot
    confidently propose somebody else's product. So ambiguity is now a
    first-class outcome instead of being collapsed into `first()`:

      client   the SAGE row used for the name (fully compatible with the old
               behaviour: the first candidate of the best tier)
      code     CodigoCliente, ONLY when unambiguous; None when in doubt
      unique   whether `code` can be trusted to read the order history

    Operating rule: without `unique`, the history layer stays off and the bot
    falls back to the catalog. We would rather ask than be accidentally right.

    SAGE data notes: CodigoCliente is zero-padded ('000139'), so string
    comparison only, never an int cast. The same client appears once per
    CodigoEmpresa — several rows with the SAME CodigoCliente are not ambiguity;
    several with different codes are.
    """

    client: Optional[Any] = None
    code: Optional[str] = None
    unique: bool = False
    source: str = "none"                  # sim | sage | none
    match_kind: str = "none"              # suffix | contains | email | sim | none
    candidate_codes: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return (getattr(self.client, "Nombre", "") or "").strip()

    @property
    def found(self) -> bool:
        return self.client is not None

    @property
    def history_allowed(self) -> bool:
        """The history layer only turns on for an unambiguous CodigoCliente."""
        return bool(self.code) and self.unique

    def describe(self) -> str:
        return (f"source={self.source} kind={self.match_kind} unique={self.unique} "
                f"code={self.code or '-'} candidates={self.candidate_codes}")


def _sim_identity(obj: Any) -> ClientIdentity:
    code = (getattr(obj, "CodigoCliente", "") or "").strip() or None
    return ClientIdentity(client=obj, code=code, unique=bool(code),
                          source="sim", match_kind="sim",
                          candidate_codes=[code] if code else [])


def _strip_punct_sql(col):
    """Nested REPLACE to drop punctuation server-side.

    Required because the WHERE clause must be a SUPERSET of what we later
    filter in Python: if SQL does not normalize, '+34 600-123.456' never
    contains '600123456' and the row never reaches the tie-break.
    """
    expr = col
    for ch in _PHONE_PUNCT:
        expr = func.replace(expr, ch, "")
    return expr


def _pick_identity(rows: List[Any], kind: str) -> ClientIdentity:
    """Build the identity from one tier's rows."""
    if not rows:
        return ClientIdentity()
    codes: List[str] = []
    for r in rows:
        code = (getattr(r, "CodigoCliente", "") or "").strip()
        if code and code not in codes:
            codes.append(code)
    unique = len(codes) == 1
    return ClientIdentity(client=rows[0], code=codes[0] if unique else None,
                          unique=unique, source="sage", match_kind=kind,
                          candidate_codes=codes)


async def resolve_client_by_phone(phone: str) -> ClientIdentity:
    term = _norm_phone(phone)
    if len(term) < 9:
        return ClientIdentity()

    # 2) SAGE (source of truth). ONE query with the widest filter (contains);
    #    the tier tie-break happens in Python, where it is testable.
    #    ORDER BY is not optional: `.limit()` becomes TOP, and a TOP without
    #    ORDER BY returns ARBITRARY rows — ambiguity detection would then be
    #    nondeterministic and two runs could disagree.
    async with AsyncSessionSQL() as s:
        cols = (MSClient.Telefono, MSClient.Telefono2, MSClient.Telefono3)
        stmt = (
            select(MSClient)
            .where(or_(*(_strip_punct_sql(c).like(f"%{term}%") for c in cols)))
            .order_by(MSClient.CodigoCliente, MSClient.CodigoEmpresa)
            .limit(_MAX_CANDIDATE_ROWS)
        )
        rows = list((await s.execute(stmt)).scalars().all())

    if not rows:
        return ClientIdentity()

    # Two match tiers, and they are never mixed — mixing them is what produced
    # the false positive:
    #   suffix    the stored number ENDS in those 9 digits. Covers '600123456',
    #             '+34600123456', '34 600-123-456'. The real case.
    #   contains  the digits appear mid-string. Covers junk like
    #             '600123456 (movil)' but also longer numbers that happen to
    #             contain those digits.
    suffix_rows, contains_rows = [], []
    for r in rows:
        values = [_only_digits(getattr(r, c, "") or "")
                  for c in ("Telefono", "Telefono2", "Telefono3")]
        if any(v.endswith(term) for v in values if v):
            suffix_rows.append(r)
        elif any(term in v for v in values if v):
            contains_rows.append(r)

    ident = (_pick_identity(suffix_rows, "suffix") if suffix_rows
             else _pick_identity(contains_rows, "contains"))
    if not ident.found:
        # Came through the server-side LIKE but does not survive Python
        # normalization (e.g. the digits matched some other column).
        return ClientIdentity()
    if not ident.unique:
        logger.warning(f"[MSG_FLOW] Phone {phone} is AMBIGUOUS in SAGE "
                       f"({ident.match_kind}): {ident.candidate_codes}. "
                       f"Order history DISABLED for this conversation.")
    return ident


async def resolve_client_by_email(email: str) -> ClientIdentity:
    addr = (email or "").strip().lower()       # 2) SAGE
    if "@" not in addr:
        return ClientIdentity()
    async with AsyncSessionSQL() as s:
        stmt = (
            select(MSClient)
            .where(or_(func.lower(MSClient.EMail1) == addr,
                       func.lower(MSClient.EMail2) == addr))
            .order_by(MSClient.CodigoCliente, MSClient.CodigoEmpresa)
            .limit(_MAX_CANDIDATE_ROWS)
        )
        rows = list((await s.execute(stmt)).scalars().all())

    ident = _pick_identity(rows, "email")
    if ident.found and not ident.unique:
        logger.warning(f"[MSG_FLOW] Email {email} is AMBIGUOUS in SAGE: "
                       f"{ident.candidate_codes}. Order history DISABLED.")
    return ident


async def resolve_client_identity(identifier: str) -> ClientIdentity:
    """Convenience: pick the lookup by the shape of the identifier."""
    return await (resolve_client_by_email(identifier) if "@" in (identifier or "")
                  else resolve_client_by_phone(identifier))


# These two keep returning the row (or None), so the ingestion gatekeepers and
# handle_ai_trigger are byte-for-byte unaffected. The NEW information — whether
# the CodigoCliente is unambiguous — comes from resolve_client_* above, which is
# what the history layer will use.
async def _find_client_by_phone(phone: str) -> Optional[Any]:
    return (await resolve_client_by_phone(phone)).client


async def _find_client_by_email(email: str) -> Optional[Any]:
    return (await resolve_client_by_email(email)).client


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


def _history_lines(history: List[Any], last: int = 10) -> str:
    lines = []
    for h in history[-last:]:
        sender = "Asistente" if h.is_bot else ("Comercial" if h.direction == "sent" else "Cliente")
        lines.append(f"{sender}: {(h.message or '').replace(chr(10), ' ').strip()}")
    return "\n".join(lines)


# --------------------------------------------------------------- admin flows
def _send_system_wa(to_phone: str, text: str, from_jid: Optional[str] = None) -> Dict[str, Any]:
    """Admin help and login/logout replies, from the device that received the command."""
    return whatsapp_transport.send_message(to_phone=to_phone, text=text, from_jid=from_jid)


def _start_mailbox(user: User) -> bool:
    """Start the Gmail listener for a salesman, resuming from his stored IMAP cursor.
    MOCK_EMAIL lets a developer route one user's mailbox through the SMTP account."""
    login_email = (user.email or "").strip().lower()
    mock_email = (os.getenv("MOCK_EMAIL") or "").strip().lower()
    smtp_user = (os.getenv("SMTP_USER") or "").strip().lower()
    imap_login = smtp_user if (mock_email and smtp_user and login_email == mock_email) else login_email
    pwd = email_transport.get_app_password(imap_login)
    if not imap_login or not pwd:
        return False
    return email_transport.start_mailbox(mailbox_email=imap_login, mailbox_password=pwd,
                                         user_mailbox=imap_login, last_uids=user.imap_uids or {})


async def login_user(user: User) -> Dict[str, Any]:
    """Pair the salesman's WhatsApp device and start his mailbox listener."""
    login_email = (user.email or "").strip().lower()
    if not login_email:
        return {"success": False, "error": "user_email_missing"}
    email_monitoring = _start_mailbox(user)
    login_resp = whatsapp_transport.start_login(phone_number=(user.phone or "").strip())
    code = login_resp.get("code") or ""
    if not login_resp.get("success"):
        return {"success": False, "error": login_resp.get("error") or "start_login_failed"}
    if code:
        email_transport.send_pairing_code_email(to_email=login_email, name=user.name or "User", code=code)
    async with AsyncSessionPG() as db:
        db_user = (await db.execute(select(User).where(User.id == user.id))).scalars().first()
        if db_user:
            db_user.enabled = True
        # MOCK_EMAIL: the SMTP account stands in for this user's mailbox.
        mock, smtp_user = (os.getenv("MOCK_EMAIL") or "").strip().lower(), (os.getenv("SMTP_USER") or "").strip().lower()
        if mock and smtp_user and login_email == mock:
            u_mail = (await db.execute(select(User).where(func.lower(User.email) == smtp_user))).scalars().first()
            if not u_mail:
                db.add(User(name="Admin SMTP", email=smtp_user, phone="", role="user", enabled=True))
            else:
                u_mail.enabled = True
        await db.commit()
    return {"success": True, "email_monitoring": email_monitoring, "code": code}


async def logout_user(user: User) -> Dict[str, Any]:
    if user.email:
        email_transport.stop_mailbox(user.email)
    if user.phone:
        try:
            whatsapp_transport.logout_device(user.phone)
            whatsapp_transport.delete_device(user.phone)
        except Exception as e:   # meow down: still disable the user
            logger.warning(f"WhatsApp logout for {user.phone} failed: {e}")
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
        _send_system_wa(
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

    _send_system_wa(to_phone=phone, text=reply, from_jid=reply_from_jid)
    logger.info(f"Admin cmd {cmd} for {user.email} (reply via {reply_from_jid or 'default'}): {out}")


# ------------------------------------------------------------ WhatsApp ingest
async def handle_new_message(payload: Dict[str, Any]) -> None:
    """Store one WhatsApp message. Answering is the loop's job (logic/loop.py)."""
    _prune_caches()
    msg = payload.get("normalized")
    if not msg:
        return
    from_phone: str = getattr(msg, "from_phone", "") or ""
    to_phone: str = getattr(msg, "to_phone", "") or ""
    text_msg: str = (getattr(msg, "text", "") or "").strip()
    filename: str = getattr(msg, "filename", "") or ""
    binary: bytes = getattr(msg, "binary", b"") or b""

    if from_phone == "__DEVICE__":
        await handle_device_event(to_phone, text_msg)
        return

    async with AsyncSessionPG() as db:
        user_from = await _get_user_by_phone(db, from_phone)
        user_to = await _get_user_by_phone(db, to_phone)

        # Self-chat = a linked salesman messaging his own number: he plays the
        # client (the live check before a release).
        is_self_chat = bool(_norm_phone(from_phone)) and _norm_phone(from_phone) == _norm_phone(to_phone)
        is_mock_owner = is_self_chat and user_from is not None

        if is_mock_owner:
            direction, user_phone, client_phone = "received", to_phone, from_phone
            device_jid, internal_user = getattr(msg, "to_jid", None), (user_to or user_from)
        elif user_from:  # the salesman wrote (meow never echoes the bot's own sends)
            direction, user_phone, client_phone = "sent", from_phone, to_phone
            device_jid, internal_user = getattr(msg, "from_jid", None), user_from
        elif user_to:    # a client wrote to a salesman
            direction, user_phone, client_phone = "received", to_phone, from_phone
            device_jid, internal_user = getattr(msg, "to_jid", None), user_to
        else:
            return

        # Admin commands: login/logout for the SENDER's own account, never the AI.
        # The reply goes from the device that received it back to the sender.
        admin_cmd = ADMIN_CMD_RE.match(text_msg)
        to_is_admin = bool(user_to and user_to.role == "admin")
        if admin_cmd and (is_mock_owner or internal_user.role == "admin" or to_is_admin):
            await event_bus.emit("admin_command", {
                "command": admin_cmd.group(1).lower(), "phone": from_phone,
                "reply_from_jid": getattr(msg, "to_jid", None)})
            return
        if to_is_admin and not is_mock_owner:
            if user_from and "ChatLink Admin Help" not in text_msg:
                now, cache_key = _now_utc(), f"{from_phone}_{to_phone}"
                last = _admin_help_cache.get(cache_key)
                if not last or (now - last).total_seconds() > 10:
                    _admin_help_cache[cache_key] = now
                    _send_system_wa(to_phone=from_phone, text=ADMIN_HELP_TEXT,
                                    from_jid=getattr(msg, "to_jid", None))
            return

        if not internal_user.enabled and not is_mock_owner:
            return

        # Gatekeeper: only SAGE clients are stored. If SAGE is down the message
        # is stored anyway (unanswered until SAGE is back) instead of lost.
        if direction == "received" and not is_mock_owner:
            try:
                if not await _find_client_by_phone(client_phone):
                    logger.info(f"[MSG_FLOW] WA dropped (not a SAGE client): {client_phone}")
                    return
            except Exception as e:
                alert("sage", f"SAGE lookup failed: {e!r}. Messages are stored but Kapa cannot answer.")

        if device_jid:
            internal_user.wa_device_jid = device_jid

        inferred = _infer_input_type(filename, "", binary)
        final_text = text_msg
        if binary:
            try:
                extracted = await _extract_media_text(binary, filename, inferred)
                prefix = _MEDIA_PREFIX.get(inferred, "[Documento]:")
                final_text = (final_text + "\n" if final_text else "") + f"{prefix} {extracted}"
            except Exception as e:
                logger.warning(f"Media parsing failed: {e}")

        # Transport retries deliver the same message twice within seconds.
        cutoff = _now_utc() - timedelta(seconds=10)
        if (await db.execute(select(Chat.id).where(
                Chat.chat_id == client_phone, Chat.user == user_phone, Chat.direction == direction,
                Chat.message == final_text, Chat.timestamp >= cutoff).limit(1))).scalars().first():
            return

        db.add(Chat(chat_id=client_phone, user=user_phone, client=client_phone, message=final_text,
                    direction=direction, input_type=inferred if binary else InputType.TEXT,
                    is_bot=False, timestamp=_now_utc()))
        await db.commit()


async def handle_device_event(phone: str, kind: str) -> None:
    """meow reports a salesman's device changed state (A7)."""
    async with AsyncSessionPG() as db:
        user = await _get_user_by_phone(db, phone)
        name = getattr(user, "name", None) or phone
        if user and kind in ("logged_out", "stream_replaced", "temporary_ban"):
            user.enabled = False
            user.wa_device_jid = None
            await db.commit()
    why = {"logged_out": "WhatsApp unlinked the device",
           "stream_replaced": "another session replaced the device's connection",
           "temporary_ban": "WhatsApp temporarily banned the number",
           "client_outdated": "WhatsApp says the client is outdated: bump whatsmeow and redeploy"}.get(kind, kind)
    tail = "" if kind == "client_outdated" else \
        " Kapa no longer covers this salesman: send `login` to the admin number to link again."
    alert(f"device:{phone}:{kind}", f"{name} ({phone}): {why}.{tail}",
          also_to=[] if kind == "client_outdated" else [getattr(user, "email", "")])


# --------------------------------------------------------------- Email ingest
async def handle_new_email(payload: Dict[str, Any]) -> None:
    """Store one email. Answering is the loop's job."""
    user_mailbox = (payload.get("user_mailbox") or "").strip().lower()
    from_email = (payload.get("from") or "").strip().lower()
    to_email = (payload.get("to") or "").strip().lower()
    subject = (payload.get("subject") or "").strip()
    body = (payload.get("body") or "").strip()
    attachments: List[Dict[str, Any]] = payload.get("attachments") or []
    direction_hint = (payload.get("direction") or "").strip().lower()
    if not user_mailbox:
        return

    async with AsyncSessionPG() as db:
        user = await _get_user_by_email(db, user_mailbox)
        if not user:
            return
        if payload.get("folder") and payload.get("uid"):
            uids = dict(user.imap_uids or {})
            uids[payload["folder"]] = max(int(uids.get(payload["folder"], 0)), int(payload["uid"]))
            user.imap_uids = uids
            await db.commit()
        if not user.enabled:
            return

        is_self_email = from_email == user_mailbox and to_email == user_mailbox
        if is_self_email:
            direction = "received"
        elif direction_hint in ("sent", "received"):
            direction = direction_hint
        else:
            direction = "sent" if from_email == user_mailbox else "received"
        client_email = (to_email if direction == "sent" else from_email).strip().lower()

        if direction == "received" and not is_self_email:
            try:
                if not await _find_client_by_email(client_email):
                    return
            except Exception as e:
                alert("sage", f"SAGE lookup failed: {e!r}. Messages are stored but Kapa cannot answer.")

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

        cutoff = _now_utc() - timedelta(minutes=int(os.getenv("EMAIL_DEDUPE_WINDOW_MINUTES", "30")))
        if (await db.execute(select(EmailChat.id).where(
                EmailChat.user == user_mailbox, EmailChat.client == client_email,
                EmailChat.direction == direction, EmailChat.message == msg_text,
                EmailChat.timestamp >= cutoff).limit(1))).scalars().first():
            return

        db.add(EmailChat(chat_id=client_email, user=user_mailbox, client=client_email, message=msg_text,
                         direction=direction, input_type=input_type, is_bot=False, timestamp=_now_utc()))
        await db.commit()


# -------------------------------------------------------------------- RAG glue


def _history_hit(m, fuente: str = "pedidos") -> Dict[str, Any]:
    """One history match in the shape the agent prompt expects.

    Only code, name and brand cross over (agent_view()). The order statistics
    stay in code: the history resolves WHICH product is meant, it does not give
    the bot the ability to discuss past orders.

    covers_query is True and query_uncovered empty because full token coverage
    is exactly what earned the verdict — a fact, not an assumption. The score is
    a flat 1.0 for every history hit on purpose: _query_verdict's "leading" tier
    would otherwise auto-pick the most-frequently-ordered candidate and merely
    mention the rest, i.e. the bot acting on purchase history instead of asking.
    Equal scores force >1 candidate to become a question, and 1.0 stays under
    the 2.0 exact-code short-circuit.
    """
    view = m.agent_view()
    return {
        "CodigoArticulo": view["codigo"],
        "DescripcionArticulo": view["nombre"],
        "MarcaProducto": view["marca"],
        "relevance_score": 1.0,
        "covers_query": True,
        "query_uncovered": [],
        "fuente": fuente,
    }


# Candidates retrieved per mention. Three was too tight for near-identical
# product families: for "adhesivo ardell transparente" the fused top-3 came back
# as one real match plus two products sharing only the word "ardell", while the
# other genuine transparent adhesives sat at rank 4-5 and never reached the
# match helper — so there was nothing left to offer as a choice. Widening costs
# a slightly larger match payload and nothing else; the triage still shows at
# most 3 options.
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

# An article the client REGULARLY BUYS whose K_BOT keeps it out of the catalog
# is offered anyway: they demonstrably buy it from this company, it appears in
# recent orders, and the salesman validates every order before it ships. The
# two alternatives both proved worse in testing — asking for a code that is not
# in the catalog either is a dead end, and falling through to the catalog is
# what committed a DIFFERENT adhesive in the first place.


async def _build_rag_candidates(queries: List[str], top_k: int = RAG_TOP_K,
                                client_code: Optional[str] = None,
                                scope: str = "auto") -> Dict[str, List[Dict[str, Any]]]:
    """Candidates for each mention, from THIS client's order history only.

    Against ~8,000 articles "el shampoo de coco" is unresolvable; against the
    40-300 this client has bought it usually resolves outright. Whether one
    candidate is taken or several become a question is decided downstream
    (llm.resolver). Every history hit scores a flat 1.0 so purchase frequency
    never silently picks a product. Near-misses ("partial") go out with
    covers_query=False, so they can only become a question.
    """
    out: Dict[str, List[Dict[str, Any]]] = {q: [] for q in queries if isinstance(q, str) and q.strip()}
    if not out or not client_code:
        if out:
            logger.info(f"[RAG] No client code: {len(out)} mention(s) can only resolve by typed code.")
        return out
    # 1) His own words for a product he picked before: an exact hit.
    async with AsyncSessionPG() as db:
        aliases = {a.alias: a for a in (await db.execute(select(ClientAlias).where(
            ClientAlias.codigo_cliente == client_code,
            ClientAlias.alias.in_([normalize_text(q) for q in out])))).scalars()}
    for q in out:
        a = aliases.get(normalize_text(q))
        if a:
            out[q] = [{"CodigoArticulo": a.code, "DescripcionArticulo": a.name, "MarcaProducto": "",
                       "relevance_score": 1.0, "covers_query": True, "query_uncovered": [],
                       "fuente": "alias"}]
    # 2) His order history.
    try:
        products, hstats = await client_history.load(client_code)
    except Exception as e:
        logger.warning(f"[RAG] History unavailable for {client_code}: {e}")
        return out
    verdicts: List[str] = []
    for q in [q for q in out if not out[q]]:
        res = client_history.match(q, products, limit=top_k)
        verdicts.append(f"{q!r}->{res.verdict}")
        if res.verdict in ("resolved", "ambiguous") and res.matches:
            out[q] = [_history_hit(m) for m in res.matches[:top_k]]
        elif res.verdict == "partial" and res.matches:
            out[q] = [{**_history_hit(m), "covers_query": False, "query_uncovered": list(m.uncovered)}
                      for m in res.matches[:top_k]]
        elif res.verdict == "blocked":
            # He buys it regularly but K_BOT hides it: offered anyway, the
            # salesman validates every order.
            out[q] = [_history_hit(m) for m in res.blocked[:top_k]]
    logger.info(f"[RAG] History {client_code} ({hstats.status}, {hstats.n_products} products): "
                + " | ".join(verdicts))
    return out


async def _sage_code_lookup(codes: List[str]) -> Dict[str, Optional[Dict[str, str]]]:
    """D4: un código TECLEADO por el cliente se acepta si existe LITERALMENTE
    en SAGE, esté o no en su historial y sea cual sea su K_BOT — verbatim del
    cliente no es ni invención ni sustitución. La línea llega al comercial
    marcada en el xlsx (columna origen) para que la revise. Un código que no
    existe en SAGE no se apunta: se dice y se deriva (lo decide la tubería)."""
    out: Dict[str, Optional[Dict[str, str]]] = {c: None for c in codes}
    if not codes:
        return out
    try:
        async with AsyncSessionSQL() as sql:
            res = await sql.execute(
                select(MSArticle).where(MSArticle.CodigoArticulo.in_(codes)))
            for a in res.scalars():
                out[str(a.CodigoArticulo)] = {
                    "code": str(a.CodigoArticulo),
                    "name": str(a.DescripcionArticulo or ""),
                    "brand": str(a.MarcaProducto or "")}
    except Exception as e:
        logger.warning(f"[AI_FLOW] code_lookup SAGE no disponible: {e}")
    return out


def _xlsx_from_cart(cart: List[Dict[str, Any]], review: Optional[List[Dict[str, Any]]] = None) -> bytes:
    """The order sheet the salesman pastes into SAGE: codigo, cantidad, origen.
    Review rows carry no code: the client's own words, for the salesman to resolve."""
    import io
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "pedido"
    ws.append(["codigo", "cantidad", "origen"])
    for item in cart:
        origen = "codigo_cliente (revisar)" if item.get("fuente") == "codigo_cliente" else ""
        ws.append([item.get("code") or "", int(item.get("qty") or 1), origen])
    for r in review or []:
        ws.append(["", int(r.get("qty") or 1), f"revisar: {r.get('texto') or ''}"])
    ws.column_dimensions["A"].width = 18
    ws.column_dimensions["B"].width = 10
    ws.column_dimensions["C"].width = 40
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ------------------------------------------------------------------ the turn
SEND_ATTEMPTS = int(os.getenv("SEND_ATTEMPTS", "3"))


def _model(channel: str):
    return Chat if channel == "whatsapp" else EmailChat


async def _salesman_of(db: AsyncSession, channel: str, user_id: str) -> Optional[User]:
    return await (_get_user_by_phone(db, user_id) if channel == "whatsapp" else _get_user_by_email(db, user_id))


async def run_turn(channel: str, client_id: str, user_id: str) -> None:
    """Answer the pending client messages of one conversation (logic/loop.py
    decided it is due). The reply, the session and a closed order are written
    in ONE transaction; send_queued()/send_orders() deliver them. Raises on
    transient failures (SAGE, inference) so the loop retries later."""
    model = _model(channel)
    async with AsyncSessionPG() as db:
        user_obj = await _salesman_of(db, channel, user_id)
        session = await _get_session(db, channel, client_id, user_id)
        history = await _load_history(db, channel, client_id, user_id, MAX_HISTORY)
        last_salesman_id = max((h.id for h in history if h.direction == "sent" and not h.is_bot), default=0)
        cursor = max(session.last_answered_id or 0, last_salesman_id)
        pending = [h for h in history if h.direction == "received" and not h.is_bot and h.id > cursor]
        if not pending:
            return
        newest_id = pending[-1].id

        async def done(why: str) -> None:
            session.last_answered_id = newest_id
            await db.commit()
            logger.info(f"[TURN] {channel}/{client_id}: {why}")

        if not user_obj or not user_obj.enabled:
            return await done("salesman disabled, not covering")
        current_message = "\n".join((h.message or "").strip() for h in pending).strip()
        if not current_message:
            return await done("nothing to answer")

        # Opt-out is respected unless the client calls Kapa by name.
        if not session.bot_enabled:
            if BOT_NAME_RE and BOT_NAME_RE.search(current_message):
                session.bot_enabled = True
            else:
                return await done("client opted out")

        identity = await resolve_client_identity(client_id)
        if not identity.found:
            if _norm_phone(client_id) == _norm_phone(user_id) or client_id == user_id:
                identity = _sim_identity(_fake_client(client_id))   # self-chat live check
            else:
                return await done("not a SAGE client")
        elif not identity.unique:
            alert(f"ambiguous:{client_id}",
                  f"{client_id} matches several SAGE clients {identity.candidate_codes}: Kapa cannot use "
                  "their order history. Fix the phone/email in SAGE.")
        client_name = (getattr(identity.client, "Nombre", "") or "").strip() or "Cliente"
        history_code = identity.code if identity.history_allowed else None

        # Lifecycle: only a CLOSED order starts a fresh cart. Silence, an
        # opt-out or a goodbye never drop the lines the client already gave.
        if session.order_status == "CLOSED" and session.cart:
            session.last_closed_cart = session.cart
        intro_mode, ended = conversation_mode(
            introduced=session.bot_introduced_at is not None,
            conv_open=bool(session.conv_open),
            order_status=session.order_status or "IDLE",
            gap_expired=_gap_expired(session))
        fresh = (session.order_status or "IDLE") == "CLOSED"
        salesman_name = getattr(user_obj, "name", "") or "Comercial"

        result: AgentResult = await run_agent(
            client_name=client_name,
            salesman_name=salesman_name,
            session={"order_status": "IDLE" if fresh else (session.order_status or "IDLE"),
                     "cart": [] if fresh else (session.cart or []),
                     "summary": "",
                     "open_items": [] if fresh else (session.open_items or []),
                     "guide_shown": False if fresh else bool(session.guide_shown),
                     "last_closed_cart": session.last_closed_cart or []},
            recent_history=_history_lines(history),
            current_message=current_message,
            intro_mode=intro_mode,
            search=partial(_build_rag_candidates, client_code=history_code),
            code_lookup=_sage_code_lookup,
            resume=ended and not fresh and bool(session.cart),
        )
        if result.ctx.get("inference_down"):
            raise RuntimeError("inference is not answering")

        # The salesman wrote while Kapa was thinking: he has it (C4).
        newer = (await db.execute(select(func.max(model.id)).where(
            model.chat_id == client_id, model.user == user_id,
            model.direction == "sent", model.is_bot.is_(False)))).scalar() or 0
        if newer > last_salesman_id:
            return await done("salesman replied during the turn; reply dropped")

        if result.opt_out:
            reply = OPTOUT_TEXT.format(salesman=salesman_name)
        elif result.handoff:
            reply = result.reply.strip() or HANDOFF_TEXT.format(salesman=salesman_name)
        else:
            reply = result.reply.strip()
        spoke_as_kapa = bool(reply) and not result.opt_out

        outcome = lifecycle_after_turn(
            intro_mode=intro_mode, spoke_as_kapa=spoke_as_kapa,
            result_status=result.order_status, cart_size=len(result.cart),
            handoff=result.handoff, opt_out=result.opt_out)
        session.order_status, session.cart = result.order_status, result.cart
        session.open_items = result.open_items
        session.guide_shown = bool(result.guide_shown)
        if result.opt_out:
            session.bot_enabled = False
        if outcome.conv_open is not None:
            session.conv_open = outcome.conv_open
        if outcome.introduced_now:
            session.bot_introduced_at = _now_utc()
        session.last_client_msg_at = pending[-1].timestamp
        session.last_answered_id = newest_id
        if reply:
            db.add(model(chat_id=client_id, user=user_id, client=client_id, message=reply,
                         direction="sent", input_type=InputType.TEXT, is_bot=True,
                         status="queued", timestamp=_now_utc()))
        if history_code:
            await _learn_aliases(db, history_code, result.learned)
        if result.order_status == "CLOSED":
            session.last_closed_cart = result.cart
            db.add(KapaOrder(channel=channel, client_id=client_id, user_id=user_id,
                             client_name=client_name, cart=result.cart,
                             review=[{"texto": i.get("texto"), "qty": i.get("qty")}
                                     for i in result.open_items if i.get("tipo") == "revisar"]))
        await db.commit()
        logger.info(f"[TURN] {channel}/{client_id}: {'replied' if reply else 'silent'}, "
                    f"status={result.order_status}, cart={len(result.cart)}")


async def _learn_aliases(db: AsyncSession, client_code: str, learned) -> None:
    """Save the client's words for the product he picked (an exact hit next time)."""
    for words, code, name in learned:
        alias = normalize_text(words)
        if not alias or not code:
            continue
        row = await db.get(ClientAlias, (client_code, alias))
        if row:
            row.code, row.name, row.uses = code, name or row.name, (row.uses or 0) + 1
        else:
            db.add(ClientAlias(codigo_cliente=client_code, alias=alias, code=code, name=name))


async def send_queued() -> None:
    """Deliver queued bot replies; after SEND_ATTEMPTS failures, alert."""
    async with AsyncSessionPG() as db:
        for channel in ("whatsapp", "email"):
            model = _model(channel)
            rows = (await db.execute(select(model).where(
                model.is_bot.is_(True), model.status == "queued").order_by(model.id))).scalars().all()
            for row in rows:
                user = await _salesman_of(db, channel, row.user)
                if channel == "whatsapp":
                    res = await asyncio.to_thread(whatsapp_transport.send_message, to_phone=row.client,
                                                  text=row.message, from_jid=getattr(user, "wa_device_jid", None))
                    ok, err = bool(res.get("success")), res.get("error")
                else:
                    ok, err = await asyncio.to_thread(email_transport.send_email_as, from_email=row.user,
                                                      to_email=row.client, subject="Re: Pedido", body=row.message)
                row.attempts = (row.attempts or 0) + 1
                if ok:
                    row.status = "sent"
                elif row.attempts >= SEND_ATTEMPTS:
                    row.status = "failed"
                    alert(f"send:{channel}:{row.user}",
                          f"A reply to {row.client} could not be delivered from {row.user} ({channel}): {err}. "
                          "The client has no answer; the salesman should reply.",
                          also_to=[getattr(user, "email", "")])
                await db.commit()


async def send_orders() -> None:
    """Email every closed order to its salesman as pedido.xlsx, with retries."""
    async with AsyncSessionPG() as db:
        orders = (await db.execute(select(KapaOrder).where(
            KapaOrder.email_status == "queued").order_by(KapaOrder.id))).scalars().all()
        for o in orders:
            user = await _salesman_of(db, o.channel, o.user_id)
            to = getattr(user, "email", None)
            if not to:
                ok, err = False, "the salesman has no email"
            else:
                lines = "\n".join(f"  {i.get('code')}  x{i.get('qty')}  {i.get('name') or ''}" for i in o.cart)
                review = "\n".join(f"  REVISAR: {r.get('texto')}" + (f" x{r['qty']}" if r.get("qty") else "")
                                   for r in o.review)
                ok, err = await asyncio.to_thread(
                    email_transport.send_email, to_email=to,
                    subject=f"Pedido confirmado ({o.client_name or o.client_id})",
                    body=f"Pedido confirmado por {o.client_name} ({o.client_id}).\n\n{lines}\n{review}".rstrip(),
                    attachments=[("pedido.xlsx",
                                  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                  _xlsx_from_cart(o.cart, o.review))])
            o.attempts = (o.attempts or 0) + 1
            if ok:
                o.email_status = "sent"
            elif o.attempts >= SEND_ATTEMPTS:
                o.email_status = "failed"
                alert(f"order:{o.id}", f"Closed order {o.id} of {o.client_name} ({o.client_id}) could not be "
                      f"emailed to {to or o.user_id}: {err}. Items: {o.cart} Review: {o.review}")
            await db.commit()


async def resume_mailboxes() -> None:
    """Restart the Gmail listeners of enabled salesmen after a restart."""
    async with AsyncSessionPG() as db:
        users = (await db.execute(select(User).where(User.enabled.is_(True)))).scalars().all()
    for u in users:
        _start_mailbox(u)
