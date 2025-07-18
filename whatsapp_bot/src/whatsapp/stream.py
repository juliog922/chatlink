import os
import logging
import grpc
from datetime import datetime
from sqlalchemy.orm import Session

from src.proto.whatsapp_pb2 import Empty, MessageEvent
from src.core.database import get_sqlserver_session, get_sqlite_session
from src.grpc.handlers import send_message, delete_device, login_and_send_qr
from src.ai.agent import handle_incoming_message
from src.media.ocr import extract_text_from_image
from src.media.audio import transcribe_audio
from src.media.documents import (
    extract_text_from_csv,
    extract_text_from_docx,
    extract_text_from_pdf,
    extract_text_from_txt,
    extract_text_from_xlsx,
)
from src.models.user import User
from src.models.message import Message
from src.models.client import Cliente


def normalize_number(raw):
    return raw.split(":")[0].lstrip("+")


def handle_admin_command(
    msg: MessageEvent,
    sender_norm: str,
    receiver_norm: str,
    stub,
    sqlite_session: Session,
):
    admins = [admin.phone for admin in User.get_admins(sqlite_session)]

    is_to_admin = any(receiver_norm.endswith(admin) for admin in admins)
    is_user_self = sender_norm == receiver_norm

    if not is_to_admin or is_user_self:
        return False

    message_text = msg.text.lower().strip()

    if message_text.startswith("logout"):
        user = User.get_by_phone(sqlite_session, sender_norm)
        if user:
            logging.info(f"Logout requested by {sender_norm}")
            delete_device(stub, sender_norm)
        else:
            logging.warning(f"Invalid logout attempt from {sender_norm}")
        return True

    elif message_text.startswith("login"):
        user = User.get_by_phone(sqlite_session, sender_norm)
        if user:
            logging.info(f"Login requested by {sender_norm}")
            login_and_send_qr(stub, sender_norm)
        else:
            logging.warning(f"Invalid login attempt from {sender_norm}")
        return True
    
    elif message_text.startswith("register"):
        # Esperado: register <name> <phone> <email>
        parts = msg.text.strip().split(maxsplit=3)
        if len(parts) != 4:
            send_message(
                stub,
                sender_norm,
                "❌ Invalid format. Use:\n`register <name> <phone> <email>`",
                receiver_norm,
            )
            return True

        _, name, phone, email = parts

        # Validación básica
        if not phone.isdigit() or "@" not in email:
            send_message(
                stub,
                sender_norm,
                "❌ Invalid phone or email format.",
                receiver_norm,
            )
            return True

        # Verificar si ya existe el teléfono o el email
        phone_exists = sqlite_session.query(User).filter_by(phone=phone).first()
        email_exists = sqlite_session.query(User).filter_by(email=email).first()

        if phone_exists or email_exists:
            send_message(
                stub,
                sender_norm,
                "⚠️ Un usuario con este email o telefono ya existe.",
                receiver_norm,
            )
            return True

        # Insertar nuevo usuario
        new_user = User(phone=phone, email=email, name=name, role="user")
        sqlite_session.add(new_user)
        sqlite_session.commit()

        send_message(
            stub,
            sender_norm,
            f"✅ User `{name}` registrado exitosamente!",
            receiver_norm,
        )
        return True

    if User.user_exists(sqlite_session, sender_norm):
        help_text = (
            "📋 *Comandos disponibles:*\n\n"
            "🔐 `login`\n"
            "Inicia sesion y envia QR al email.\n\n"
            "🚪 `logout`\n"
            "Finaliza tu sesion.\n\n"
            "📝 `register <name> <phone> <email>`\n"
            "Registra un nuevo usuario."
        )
        send_message(stub, sender_norm, help_text, receiver_norm)

    return True


def stream_messages(stub):
    logging.info("Connecting to WhatsApp message stream...")
    base_dir = "media"
    os.makedirs(base_dir, exist_ok=True)

    sqlserver_session = get_sqlserver_session()
    sqlite_session = get_sqlite_session()

    try:

        for msg in stub.StreamMessages(Empty()):
            sender = getattr(msg, "from").split("@")[0].split(":")[0]
            receiver = msg.to.split(":")[0]
            sender_norm = normalize_number(sender)
            receiver_norm = normalize_number(receiver)

            logging.info(f"New message: {sender} → {receiver} ({msg.timestamp})")

            if handle_admin_command(
                msg, sender_norm, receiver_norm, stub, sqlite_session
            ):
                continue

            store_message_if_applicable(
                msg, sender, receiver, sqlite_session, sqlserver_session, base_dir
            )

            if msg.text.strip():
                logging.info(f"Message content: {msg.text.strip()}")
            elif msg.binary:
                logging.info(
                    f"Binary message received with filename: {msg.filename or 'unnamed_file'}"
                )

    except grpc.RpcError as e:
        logging.error(f"gRPC stream error: {e.code().name} - {e.details()}")

    finally:
        sqlserver_session.close()
        sqlite_session.close()


def store_message_if_applicable(
    msg,
    sender,
    receiver,
    sqlite_session: Session,
    sqlserver_session: Session,
    base_dir: str,
):
    direction = None
    matched_cliente = Cliente.get_by_telefono(sqlserver_session, sender)
    direction = "received" if matched_cliente else None

    if not matched_cliente:
        matched_cliente = Cliente.get_by_telefono(sqlserver_session, receiver)
        direction = "sent" if matched_cliente else None

    if not matched_cliente:
        return None, None, None

    matched_id = matched_cliente.codigo_cliente
    message_type = "text"
    content = msg.text

    # Buscar el user por teléfono
    user = User.get_by_phone(sqlite_session, sender) or User.get_by_phone(
        sqlite_session, receiver
    )

    if msg.binary:
        message_type = "media"
        filename = msg.filename or f"file_{msg.timestamp}.bin"
        ext = os.path.splitext(filename)[1].lower()

        subdir = (
            "images"
            if ext in [".jpg", ".jpeg", ".png", ".webp"]
            else (
                "audio"
                if ext in [".mp3", ".ogg", ".wav", ".opus"]
                else "video" if ext in [".mp4", ".avi", ".mkv"] else "documents"
            )
        )

        full_dir = os.path.join(base_dir, subdir)
        os.makedirs(full_dir, exist_ok=True)
        file_path = os.path.join(full_dir, filename)

        try:
            with open(file_path, "wb") as f:
                f.write(msg.binary)
            logging.info(f"Saved media file: {file_path}")

            if subdir == "images":
                text = extract_text_from_image(msg.binary)
                if text:
                    content = text
                    message_type = "text"
            elif subdir == "audio":
                text = transcribe_audio(msg.binary, extension=ext)
                if text:
                    content = text
                    message_type = "text"
            elif subdir == "documents":
                text = None
                if ext == ".pdf":
                    text = extract_text_from_pdf(file_path)
                elif ext == ".docx":
                    text = extract_text_from_docx(file_path)
                elif ext == ".txt":
                    text = extract_text_from_txt(file_path)
                elif ext == ".csv":
                    text = extract_text_from_csv(file_path)
                elif ext == ".xlsx":
                    text = extract_text_from_xlsx(file_path)

                if text and text.strip():
                    content = text.strip()
                    message_type = "text"
        except Exception as e:
            logging.error(f"Error saving media: {e}")

    Message.create(
        session=sqlite_session,
        client_id=matched_id,
        client_phone=receiver if direction == "sent" else sender,
        direction=direction,
        type_=message_type,
        content=content.replace("\n", " \\"),
        user_id=user.id,
        user_phone=sender if direction == "sent" else receiver,
        timestamp=parse_flexible_timestamp(msg.timestamp),
    )
    return matched_id, direction, message_type


def parse_flexible_timestamp(ts: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d_%H%M%S"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    raise ValueError(f"Formato de timestamp desconocido: {ts}")


def handle_ai_response_if_applicable(
    msg,
    sender,
    receiver,
    receiver_norm,
    sqlserver_session: Session,
    sqlite_session: Session,
    stub,
):
    cliente = Cliente.get_by_telefono(sqlserver_session, sender)
    if not cliente:
        return

    if not msg.text.strip():
        return

    user = User.get_by_phone(sqlite_session, receiver_norm)
    if not user:
        logging.info(f"Ignored message from {sender}: no target user")
        return

    handle_incoming_message(
        sqlite_session,
        sqlserver_session,
        stub,
        send_message,
        receiver,
        sender,
        msg.text.strip(),
    )
