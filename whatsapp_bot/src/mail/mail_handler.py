import os
import smtplib
import mimetypes
from dotenv import load_dotenv, dotenv_values
import logging
from datetime import datetime
from email.message import EmailMessage
from email.utils import formataddr

from src.models.user import User
from src.models.client import Cliente


load_dotenv()
config = dotenv_values(".env")
SMTP_SERVER = config.get("SMTP_SERVER")
SMTP_PORT = int(config.get("SMTP_PORT", 587))
SMTP_USER = config.get("SMTP_USER")
SMTP_PASSWORD = config.get("SMTP_PASSWORD")
SENDER_NAME = config.get("SENDER_NAME", "Kapalua Bot Asistant")


def send_email(recipient: str, subject: str, body: str, attachments: list[str] = None):

    msg = EmailMessage()
    msg["From"] = formataddr((SENDER_NAME, SMTP_USER))
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)

    attachments = attachments or []
    for path in attachments:
        ctype, encoding = mimetypes.guess_type(path)
        if ctype is None or encoding is not None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)

        with open(path, "rb") as f:
            file_data = f.read()
            file_name = os.path.basename(path)
            msg.add_attachment(
                file_data, maintype=maintype, subtype=subtype, filename=file_name
            )

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)
        print(f"Correo enviado a {recipient}")


# Template 1: Enviar QR para WhatsApp


def send_qr_email(recipient: str, qr_image_path: str):
    subject = "Escanea el código QR para vincular tu WhatsApp"
    body = (
        "Hola!\n\n"
        "Por favor escanea el código QR adjunto con tu aplicación de WhatsApp.\n"
        "Este código te permitirá vincular tu dispositivo para continuar usando el servicio.\n\n"
        "Pasos sugeridos:\n"
        "1. Abre WhatsApp en tu móvil\n"
        "2. Toca en los tres puntos > Dispositivos vinculados\n"
        "3. Escanea el QR de esta imagen\n"
        "\nSaludos,\nKapalua Bot Asistant"
    )
    send_email(recipient, subject, body, attachments=[qr_image_path])


# Template 2: Enviar uno o más documentos (por ahora Excel)


def send_documents_email(recipient: str, doc_paths: list[str]):
    subject = "Documentos solicitados"
    body = (
        "Hola!\n\n"
        "Adjuntamos los documentos solicitados.\n"
        "Revisa el contenido y responde si necesitas algo más.\n\n"
        "Saludos,\nKapalua Bot Asistant"
    )
    send_email(recipient, subject, body, attachments=doc_paths)


def notify_order_by_email(user: User, client: Cliente, phone: str, csv_path: str):
    if not user.email:
        logging.warning(f"⚠️ El comercial {user.name} no tiene email configurado.")
        return

    asunto = f"Nuevo pedido confirmado de {client.razon_social or 'cliente'} - {datetime.now()}"
    cuerpo = f"""Hola {user.name},

    Has recibido un nuevo pedido confirmado por parte de:

    - Cliente: {client.razon_social or 'Sin nombre registrado'}  
    - Teléfono: {phone}

    Adjunto encontrarás el detalle del pedido en formato CSV.

    Por favor, revisa el archivo y continúa con el proceso correspondiente.

    Un saludo,
    El asistente automático"""

    send_email(
        recipient=user.email, subject=asunto, body=cuerpo, attachments=[csv_path]
    )


# Ejemplo de .env necesario:
# SMTP_SERVER=smtp.gmail.com
# SMTP_PORT=587
# SMTP_USER=tu_correo@gmail.com
# SMTP_PASSWORD=tu_contraseña
# SENDER_NAME=Asistente Julio
