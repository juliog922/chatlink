from dotenv import load_dotenv
from src.config.logging_setup import setup_logging
from src.cli.parser import build_parser
from src.grpc.client import create_grpc_stub
from src.whatsapp.stream import stream_messages
from src.ai.agent import process_unattended_messages_loop
from src.grpc.handlers import (
    login,
    login_and_send_qr,
    list_devices,
    send_message,
    send_file,
    delete_device,
    login_and_send_qr_to_all_admins,
)
import threading


def main():
    load_dotenv()
    setup_logging()

    parser = build_parser()
    args = parser.parse_args()

    stub = create_grpc_stub()

    if args.cmd == "login":
        login(stub)
    elif args.cmd == "loginqr":
        login_and_send_qr(stub, args.to)
    elif args.cmd == "loginqr_all":
        login_and_send_qr_to_all_admins(stub)
    elif args.cmd == "list":
        list_devices(stub)
    elif args.cmd == "listen":
        ai_thread = threading.Thread(
            target=process_unattended_messages_loop, args=(stub,), daemon=True
        )
        ai_thread.start()

        # Iniciar escucha principal de mensajes
        stream_messages(stub)
    elif args.cmd == "send":
        send_message(stub, args.to, args.text, from_jid=args.from_jid)
    elif args.cmd == "sendfile":
        send_file(stub, args.to, args.file, from_jid=args.from_jid)
    elif args.cmd == "delete":
        delete_device(stub, args.jid)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
