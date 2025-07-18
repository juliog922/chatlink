import argparse


def build_parser():
    parser = argparse.ArgumentParser(description="WhatsApp gRPC client")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    subparsers.add_parser("login", help="Start login flow with QR code")
    subparsers.add_parser("listen", help="Start streaming incoming messages")
    subparsers.add_parser("list", help="List all registered devices")

    send_parser = subparsers.add_parser("send", help="Send a text message")
    send_parser.add_argument("--to", required=True, help="Recipient phone number")
    send_parser.add_argument("--text", required=True, help="Message text")
    send_parser.add_argument("--from", dest="from_jid", help="Device JID (optional)")

    file_parser = subparsers.add_parser("sendfile", help="Send a file")
    file_parser.add_argument("--to", required=True, help="Recipient phone number")
    file_parser.add_argument("--file", required=True, help="Path to the file")
    file_parser.add_argument("--from", dest="from_jid", help="Device JID (optional)")

    delete_parser = subparsers.add_parser("delete", help="Delete a device")
    delete_parser.add_argument("--jid", required=True, help="Device JID to remove")

    loginqr_parser = subparsers.add_parser(
        "loginqr", help="Send QR code to phone via WhatsApp"
    )
    loginqr_parser.add_argument(
        "--to", required=True, help="Phone number to send QR code to"
    )

    subparsers.add_parser("loginqr_all", help="Enviar QR a todos los administradores")

    return parser
