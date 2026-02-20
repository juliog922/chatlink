import asyncio
import email as email_pkg
import imaplib
import logging
import os
import re
import smtplib
import threading
import time
from dataclasses import dataclass
from email.header import decode_header
from email.message import EmailMessage
from email.utils import formataddr
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

from ..events import event_bus

logger = logging.getLogger("Transport.Email")

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")

# "Admin Gmail" credentials (used for SMTP sending + optional default listener)
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

POLL_INTERVAL_SECONDS = int(os.getenv("IMAP_POLL_SECONDS", "10"))


def _decode_subject(raw: str) -> str:
    if not raw:
        return ""
    decoded, enc = decode_header(raw)[0]
    if isinstance(decoded, bytes):
        return decoded.decode(enc or "utf-8", errors="replace")
    return str(decoded)


def _extract_email_addr(s: str) -> str:
    s = (s or "").strip()
    if "<" in s and ">" in s:
        return s.split("<", 1)[1].split(">", 1)[0].strip()
    return s


@dataclass
class EmailAttachment:
    filename: str
    content_type: str
    data: bytes


@dataclass
class NormalizedEmail:
    user_mailbox: str           # which mailbox listener received this
    from_email: str
    to_email: str
    subject: str
    body: str
    attachments: List[EmailAttachment]

IMAP_SENT_FOLDER = os.getenv("IMAP_SENT_FOLDER", "").strip()  # optional override

class _MailboxListener:
    def __init__(self, login_email: str, mailbox_password: str, main_loop: asyncio.AbstractEventLoop, user_mailbox: str):
        self.login_email = login_email
        self.user_mailbox = user_mailbox
        self.mailbox_password = mailbox_password
        self.main_loop = main_loop
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # track last UID per folder so we ingest sent+received exactly once
        self._last_uid: Dict[str, int] = {}
        self._sent_folder: Optional[str] = None

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(f"IMAP listener started (login={self.login_email}, user_mailbox={self.user_mailbox})")

    def stop(self) -> None:
        self._running = False

    def _emit(self, payload: Dict) -> None:
        asyncio.run_coroutine_threadsafe(event_bus.emit("email_received", payload), self.main_loop)

    def _extract_mailbox_name(self, line: str) -> Optional[str]:
        # IMAP LIST lines usually end with the mailbox name (often quoted)
        m = re.search(r'"([^"]+)"\s*$', line)
        if m:
            return m.group(1)
        parts = line.split()
        if parts:
            return parts[-1].strip('"')
        return None

    def _discover_sent_folder(self, mail: imaplib.IMAP4_SSL) -> Optional[str]:
        if IMAP_SENT_FOLDER:
            return IMAP_SENT_FOLDER

        try:
            typ, data = mail.list()
            if typ != "OK" or not data:
                return None

            names: list[str] = []
            for raw in data:
                s = raw.decode(errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
                name = self._extract_mailbox_name(s)
                if name:
                    names.append(name)

                # Prefer special-use \Sent if present
                if "\\Sent" in s or "\\\\Sent" in s:
                    if name:
                        return name

            # Fallback common names (Gmail + generic)
            for cand in ("[Gmail]/Sent Mail", "[Gmail]/Sent", "Sent", "Sent Mail", "Sent Items"):
                if cand in names:
                    return cand
        except Exception:
            return None

        return None

    def _get_uidnext(self, mail: imaplib.IMAP4_SSL, folder: str) -> Optional[int]:
        try:
            typ, data = mail.status(f'"{folder}"', "(UIDNEXT)")
            if typ != "OK" or not data:
                return None
            s = data[0].decode(errors="ignore") if isinstance(data[0], (bytes, bytearray)) else str(data[0])
            m = re.search(r"UIDNEXT\s+(\d+)", s)
            return int(m.group(1)) if m else None
        except Exception:
            return None

    def _poll_folder(self, mail: imaplib.IMAP4_SSL, folder: str, direction: str) -> None:
        # Initialize last UID to "current end" so we only ingest NEW mail after listener starts
        if folder not in self._last_uid:
            uidnext = self._get_uidnext(mail, folder)
            self._last_uid[folder] = max(0, (uidnext or 1) - 1)

        last = self._last_uid[folder]

        typ, data = mail.uid("search", None, f"UID {last + 1}:*")
        if typ != "OK" or not data or not data[0]:
            return

        raw_uids = [int(x) for x in data[0].split() if x.isdigit()]
        uids = [u for u in raw_uids if u > last]   # <-- critical anti-loop filter
        if not uids:
            return

        for uid in uids:
            typ2, msg_data = mail.uid("fetch", str(uid), "(RFC822)")
            if typ2 != "OK" or not msg_data:
                continue

            for part in msg_data:
                if isinstance(part, tuple):
                    norm = self._parse_raw(part[1])
                    if norm:
                        self._emit(
                            {
                                "transport": "email",
                                "user_mailbox": norm.user_mailbox,
                                "from": norm.from_email,
                                "to": norm.to_email,
                                "subject": norm.subject,
                                "body": norm.body,
                                "attachments": [
                                    {"filename": a.filename, "content_type": a.content_type, "bytes": a.data}
                                    for a in norm.attachments
                                ],
                                # NEW:
                                "folder": folder,
                                "direction": direction,  # "received" for inbox, "sent" for sent folder
                                "login_email": self.login_email,
                            }
                        )

        logger.info(
            f"IMAP new mail: user_mailbox={self.user_mailbox} login={self.login_email} "
            f"folder={folder} direction={direction} new_uids={len(uids)}"
        )

        self._last_uid[folder] = max(self._last_uid[folder], max(uids))

    def _loop(self) -> None:
        while self._running:
            try:
                mail = imaplib.IMAP4_SSL(IMAP_SERVER)
                mail.login(self.login_email, self.mailbox_password)

                # discover sent folder once per listener
                if not self._sent_folder:
                    self._sent_folder = self._discover_sent_folder(mail)

                # INBOX = received
                mail.select("INBOX")
                self._poll_folder(mail, "INBOX", "received")

                # SENT = sent (if available)
                if self._sent_folder:
                    try:
                        mail.select(f'"{self._sent_folder}"')
                        self._poll_folder(mail, self._sent_folder, "sent")
                    except Exception as e:
                        logger.warning(f"Could not select sent folder '{self._sent_folder}' for {self.login_email}: {e}")

                mail.close()
                mail.logout()

            except Exception as e:
                logger.error(f"IMAP loop error ({self.login_email}): {e}")

            time.sleep(POLL_INTERVAL_SECONDS)

    def _parse_raw(self, raw_bytes: bytes) -> Optional[NormalizedEmail]:
        try:
            msg = email_pkg.message_from_bytes(raw_bytes)

            subject = _decode_subject(msg.get("Subject", ""))
            from_email = _extract_email_addr(msg.get("From", ""))
            to_email = _extract_email_addr(msg.get("To", ""))

            body = ""
            attachments: List[EmailAttachment] = []

            if msg.is_multipart():
                for p in msg.walk():
                    disp = str(p.get("Content-Disposition") or "")
                    ctype = p.get_content_type()

                    if "attachment" in disp.lower():
                        filename = p.get_filename() or ""
                        data = p.get_payload(decode=True) or b""
                        attachments.append(EmailAttachment(filename=filename, content_type=ctype, data=data))
                        continue

                    if ctype == "text/plain" and "attachment" not in disp.lower():
                        payload = p.get_payload(decode=True)
                        if payload:
                            body += payload.decode(errors="replace")
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body = payload.decode(errors="replace")

            return NormalizedEmail(
                user_mailbox=self.user_mailbox,
                from_email=from_email,
                to_email=to_email,
                subject=subject,
                body=body,
                attachments=attachments,
            )
        except Exception as e:
            logger.error(f"Email parse error (login={self.login_email}, user_mailbox={self.user_mailbox}): {e}")
            return None


class EmailTransport:
    """
    Notes-aligned:
      - multi-threaded IMAP listeners per user mailbox
      - emits email_received with attachments bytes for later parsers
      - SMTP sending (admin gmail for now)
    """
    def __init__(self) -> None:
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None
        self._listeners: Dict[str, _MailboxListener] = {}  # key = user_mailbox (logical)
        self._accounts_file = os.getenv("GMAIL_ACCOUNTS_FILE", "/app/gmail_accounts.json")
        self._accounts: Dict[str, str] = {}
        self._accounts_mtime: float = 0.0

    @property
    def active_mailboxes(self) -> List[str]:
        return list(self._listeners.keys())

    def start(self) -> None:
        try:
            self._main_loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.error("Cannot start EmailTransport: no running event loop.")
            return
        self._load_accounts_if_changed()

    def stop(self) -> None:
        for l in list(self._listeners.values()):
            l.stop()
        self._listeners.clear()

    def start_mailbox(
        self,
        mailbox_email: str,                      # IMAP login email
        mailbox_password: Optional[str] = None,  # IMAP app password
        user_mailbox: Optional[str] = None,      # logical mailbox for attribution (Postgres user.email)
    ) -> bool:
        if not mailbox_email:
            return False
        if not self._main_loop:
            logger.error("EmailTransport not started (no main loop).")
            return False
        
        logical = (user_mailbox or mailbox_email).strip().lower()
        if not logical:
            return False

        # avoid duplicate threads
        existing = self._listeners.get(logical)
        if existing and existing.is_running:
            return True

        if mailbox_password is None:
            mailbox_password = self.get_app_password(mailbox_email)
        if not mailbox_password:
            # missing app password can be expected (WA-only users)
            logger.warning(f"No app password found for {mailbox_email} (email monitoring disabled for {logical})")
            return False

        listener = _MailboxListener(mailbox_email, mailbox_password, self._main_loop, user_mailbox=logical)
        self._listeners[logical] = listener
        listener.start()
        return True

    def stop_mailbox(self, mailbox_email: str) -> None:
        key = (mailbox_email or "").strip().lower()
        l = self._listeners.get(key)
        if l:
            l.stop()
            self._listeners.pop(key, None)
            logger.info(f"IMAP listener stopped for {key}")

    # -------- SMTP sending (Admin Gmail) --------

    def send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        attachments: Optional[List[Tuple[str, str, bytes]]] = None,  # (filename, mime_type, bytes)
    ) -> Tuple[bool, Optional[str]]:
        if not SMTP_USER or not SMTP_PASSWORD:
            return False, "Missing SMTP credentials"

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SMTP_USER
        msg["To"] = to_email
        msg.set_content(body or "")

        if attachments:
            for filename, mime_type, data in attachments:
                maintype, subtype = (mime_type.split("/", 1) + ["octet-stream"])[:2]
                msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)

        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)
            return True, None
        except Exception as e:
            logger.error(f"SMTP error: {e}")
            return False, str(e)

    def send_qr_email(self, to_email: str, name: str, qr_data: str) -> Tuple[bool, Optional[str]]:
        try:
            import io
            import qrcode
        except Exception as e:
            return False, f"qrcode dependency missing: {e}"

        if not SMTP_USER or not SMTP_PASSWORD:
            return False, "Missing SMTP credentials"

        subject = f"WhatsApp Login for {name}"
        body = (
            f"Hello {name},\n\n"
            "Please scan the attached QR code to link your device.\n"
            "Note: This code expires quickly.\n"
        )

        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(qr_data)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        return self.send_email(
            to_email=to_email,
            subject=subject,
            body=body,
            attachments=[("login_qr.png", "image/png", png_bytes)],
        )
    
    def _load_accounts_if_changed(self) -> None:
        try:
            p = Path(self._accounts_file)
            if not p.exists():
                logger.warning(f"Gmail accounts file not found: {self._accounts_file}")
                self._accounts = {}
                return

            mtime = p.stat().st_mtime
            if mtime == self._accounts_mtime and self._accounts:
                return

            self._accounts = json.loads(p.read_text(encoding="utf-8"))
            self._accounts_mtime = mtime
            logger.info(f"Loaded {len(self._accounts)} gmail accounts from {self._accounts_file}")
        except Exception as e:
            logger.error(f"Failed loading gmail accounts file: {e}")
            self._accounts = {}

    def get_app_password(self, email_addr: str) -> Optional[str]:
        self._load_accounts_if_changed()
        return self._accounts.get(email_addr)
    
    def send_email_as(
        self,
        from_email: str,
        to_email: str,
        subject: str,
        body: str,
        attachments: Optional[List[Tuple[str, str, bytes]]] = None,
    ) -> Tuple[bool, Optional[str]]:
        pwd = self.get_app_password(from_email)
        if not pwd:
            return False, f"No app password found for {from_email}"

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email
        msg.set_content(body or "")

        if attachments:
            for filename, mime_type, data in attachments:
                maintype, subtype = (mime_type.split("/", 1) + ["octet-stream"])[:2]
                msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)

        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(from_email, pwd)
                server.send_message(msg)
            return True, None
        except Exception as e:
            logger.error(f"SMTP send_email_as error: {e}")
            return False, str(e)
        
    def has_app_password(self, email_addr: str) -> bool:
        """Checks if a password exists for the given email."""
        self._load_accounts_if_changed()
        return bool(self._accounts.get(email_addr.strip().lower()))

    def set_app_password(self, email_addr: str, password: str) -> bool:
        """Saves a new app password to the JSON file."""
        self._load_accounts_if_changed()
        clean_email = email_addr.strip().lower()
        # Remove spaces (Google app passwords are often copied with spaces)
        clean_pass = password.replace(" ", "") 
        
        self._accounts[clean_email] = clean_pass
        try:
            with open(self._accounts_file, "w") as f:
                json.dump(self._accounts, f, indent=4)
            if self._accounts_file.exists():
                self._last_mtime = self._accounts_file.stat().st_mtime
            return True
        except Exception as e:
            logger.error(f"Failed to save gmail password: {e}")
            return False




email_transport = EmailTransport()
