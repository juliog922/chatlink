import os
import logging
import qrcode

from src.core.auth import verify_credentials
from src.core.qr import show_qr_ascii
from src.mail.mail_handler import send_qr_email
from src.proto.whatsapp_pb2 import Empty, SendRequest, DeviceID
from src.core.database import get_sqlite_session
from src.models.user import User


def login(stub):
    if not verify_credentials():
        return
    logging.info("Starting login process...")
    response = stub.StartLogin(Empty())
    if response.status == "code":
        show_qr_ascii(response.code)
    elif response.status == "already_connected":
        logging.info("Session already active.")
    elif response.status != "success":
        logging.error(f"Login error: {response.status}")


def send_message(stub, to, text, from_jid=None):
    logging.info(f"Sending to={to} from_jid={from_jid}")

    # Obtener lista de dispositivos
    try:
        devices = stub.ListDevices(Empty()).devices
        device_jids = [d.jid for d in devices]
        logging.info(f"Available devices: {device_jids}")
    except Exception as e:
        logging.error(f"Error fetching device list: {e}")
        return

    # Validar que el from_jid esté en los dispositivos activos
    if from_jid:
        if from_jid not in device_jids:
            logging.error(f"from_jid {from_jid} not found in connected devices")
            return
    else:
        logging.error("from_jid is required, but none was provided")
        return

    # Enviar mensaje
    req = SendRequest(to=to, text=text, from_jid=from_jid)
    try:
        resp = stub.SendMessage(req)
        if resp.success:
            logging.info(f"Message sent to {to}")
        else:
            logging.error(f"Failed to send message: {resp.error}")
    except Exception as e:
        logging.error(f"gRPC error while sending message: {e}")


def send_file(stub, to, filepath, from_jid=None):
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        return

    with open(filepath, "rb") as f:
        binary_data = f.read()

    req = SendRequest(
        to=to,
        text=os.path.basename(filepath),
        binary=binary_data,
        filename=os.path.basename(filepath),
        from_jid=from_jid or "",
    )

    resp = stub.SendMessage(req)
    if resp.success:
        logging.info(f"File sent to {to}: {filepath}")
    else:
        logging.error(f"Failed to send file: {resp.error}")


def list_devices(stub):
    response = stub.ListDevices(Empty())
    logging.info("Registered devices:")
    for device in response.devices:
        logging.info(f"• {device.jid}")


def delete_device(stub, jid):
    resp = stub.DeleteDevice(DeviceID(jid=jid))
    if resp.success:
        logging.info(f"Device deleted: {jid}")
    else:
        logging.error(f"Failed to delete device: {resp.error}")


def login_and_send_qr(stub, to_phone: str):
    response = stub.StartLogin(Empty())

    if response.status == "code":
        # Buscar email del usuario desde SQLite
        session = get_sqlite_session()
        try:
            user = User.get_by_phone(session, to_phone)
            if not user:
                logging.warning(f"No user found with phone: {to_phone}")
                return

            qr_img = qrcode.make(response.code)
            qr_path = "/tmp/qr.jpg"
            qr_img.save(qr_path, format="JPEG")
            logging.info(f"QR saved to {qr_path}")

            send_qr_email(user.email, qr_path)

        finally:
            session.close()

    elif response.status == "already_connected":
        logging.info("Session already active.")
    else:
        logging.error(f"Login error: {response.status}")


def login_and_send_qr_to_all_admins(stub):
    response = stub.StartLogin(Empty())

    if response.status == "code":
        session = get_sqlite_session()
        try:
            admins = User.get_admins(session)
            if not admins:
                logging.warning("No admins found in the database.")
                return

            # Generar QR una sola vez
            qr_img = qrcode.make(response.code)
            qr_path = "/tmp/qr.jpg"
            qr_img.save(qr_path, format="JPEG")
            logging.info(f"QR saved to {qr_path}")

            # Enviar a todos los correos de administradores
            for admin in admins:
                logging.info(f"Sending QR to admin: {admin.email}")
                send_qr_email(admin.email, qr_path)

        finally:
            session.close()

    elif response.status == "already_connected":
        logging.info("Session already active.")
    else:
        logging.error(f"Login error: {response.status}")
