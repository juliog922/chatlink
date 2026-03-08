import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import grpc

from ..events import event_bus
from .. import whatsapp_pb2, whatsapp_pb2_grpc

logger = logging.getLogger("Transport.WhatsApp")

GRPC_HOST = os.getenv("GRPC_HOST", "meow_server")
GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))

# Admin control (comma-separated phone numbers, no +, no @s.whatsapp.net)
# Example: ADMIN_WA_NUMBERS=34685176889,34600111222
ADMIN_WA_NUMBERS = {
    x.strip() for x in os.getenv("ADMIN_WA_NUMBERS", "").split(",") if x.strip()
}


def _jid_to_phone(jid: str) -> str:
    # Example JIDs: "34685...@s.whatsapp.net" or "34685...:xx@s.whatsapp.net"
    if not jid:
        return ""
    return jid.split("@")[0].split(":")[0]


def _is_admin_sender(from_jid: str) -> bool:
    phone = _jid_to_phone(from_jid)
    return bool(phone) and phone in ADMIN_WA_NUMBERS


def _parse_admin_command(text: str) -> Optional[Dict[str, Optional[str]]]:
    t = (text or "").strip()
    if not t:
        return None
    parts = t.split()
    cmd = parts[0].lower()
    if cmd not in ("login", "logout"):
        return None
    return {"command": cmd}



@dataclass
class WhatsAppMessage:
    raw: Any
    from_jid: str
    to_jid: str
    from_phone: str
    to_phone: str
    name: str
    text: str
    timestamp: str
    binary: bytes
    filename: str


class WhatsAppTransport:
    """
    Notes-aligned wrapper:
      - owns grpc channel + stub
      - runs StreamMessages in a background thread
      - emits:
          * message_received  (normal traffic)
          * admin_command     (admin channel commands)
    """
    def __init__(self) -> None:
        self.channel: Optional[grpc.Channel] = None
        self.stub: Optional[whatsapp_pb2_grpc.WhatsAppServiceStub] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        try:
            self._main_loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.error("Cannot start WhatsAppTransport: no running event loop.")
            return

        target = f"{GRPC_HOST}:{GRPC_PORT}"
        logger.info(f"Connecting to WhatsApp gRPC at {target}...")

        self.channel = grpc.insecure_channel(target)
        self.stub = whatsapp_pb2_grpc.WhatsAppServiceStub(self.channel)
        self._running = True

        self._thread = threading.Thread(target=self._listen_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        try:
            if self.channel:
                self.channel.close()
        except Exception:
            pass

    def _emit_threadsafe(self, event_name: str, payload: Any) -> None:
        if self._main_loop and self._main_loop.is_running():
            asyncio.run_coroutine_threadsafe(event_bus.emit(event_name, payload), self._main_loop)

    def _listen_forever(self) -> None:
        logger.info("WhatsApp stream thread started.")
        backoff = 1

        while self._running:
            try:
                if not self.stub:
                    time.sleep(1)
                    continue

                stream = self.stub.StreamMessages(whatsapp_pb2.Empty())
                for ev in stream:
                    msg = WhatsAppMessage(
                        raw=ev,
                        from_jid=getattr(ev, "from", ""),
                        to_jid=getattr(ev, "to", ""),
                        from_phone=_jid_to_phone(getattr(ev, "from", "")),
                        to_phone=_jid_to_phone(getattr(ev, "to", "")),
                        name=getattr(ev, "name", "") or "",
                        text=getattr(ev, "text", "") or "",
                        timestamp=getattr(ev, "timestamp", "") or "",
                        binary=getattr(ev, "binary", b"") or b"",
                        filename=getattr(ev, "filename", "") or "",
                    )

                    # Admin command channel
                    is_admin_chat = (msg.from_phone in ADMIN_WA_NUMBERS) or (msg.to_phone in ADMIN_WA_NUMBERS)
                    if is_admin_chat:
                        parsed = _parse_admin_command(msg.text)
                        if parsed:
                            self._emit_threadsafe("admin_command", {
                                "transport": "whatsapp",
                                "phone": msg.from_phone,
                                "command": parsed["command"]
                            })
                            continue

                    # Normal ingress
                    self._emit_threadsafe("message_received", {"msg": ev, "normalized": msg})

                backoff = 1

            except grpc.RpcError as e:
                logger.warning(f"WA stream interrupted: {e.details()} (retry in {backoff}s)")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
            except Exception as e:
                logger.error(f"WA listener error: {e}")
                time.sleep(5)

    # ----- Outgoing control -----

    def start_login(self, phone_number: str) -> Dict[str, Any]:
        if not self.stub:
            return {
                "success": False,
                "status": "error",
                "code": None,
                "error": "stub_not_ready",
            }

        try:
            req = whatsapp_pb2.LoginRequest(phone_number=phone_number)
            resp = self.stub.StartLogin(req)

            status = (getattr(resp, "status", "") or "").strip()
            code = getattr(resp, "code", None)

            err = (getattr(resp, "error", "") or "").strip()
            success = status.lower() in ("ok", "success") or bool(code)

            if not success and not err:
                err = status or "start_login_failed"

            return {
                "success": success,
                "status": status,
                "code": code,
                "error": err,
            }

        except grpc.RpcError as e:
            return {"success": False, "status": "error", "code": None, "error": e.details() or str(e)}
        except Exception as e:
            return {"success": False, "status": "error", "code": None, "error": str(e)}

    def send_message(
        self,
        to_phone: str,
        text: Optional[str] = None,
        from_jid: Optional[str] = None,
        binary: Optional[bytes] = None,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.stub:
            return {"success": False, "error": "stub_not_ready"}

        req = whatsapp_pb2.SendRequest(
            to=to_phone or "",
            text=text or "",
            from_jid=from_jid or "",
            binary=binary or b"",
            filename=filename or "",
        )
        resp = self.stub.SendMessage(req)
        return {"success": bool(getattr(resp, "success", False)), "error": getattr(resp, "error", "")}

    def list_devices(self) -> List[str]:
        if not self.stub:
            return []
        resp = self.stub.ListDevices(whatsapp_pb2.Empty())
        return [d.jid for d in getattr(resp, "devices", [])]

    def logout_device(self, jid: str) -> Dict[str, Any]:
        if not self.stub:
            return {"success": False, "error": "stub_not_ready"}
        resp = self.stub.LogoutDevice(whatsapp_pb2.DeviceID(jid=jid))
        return {"success": bool(getattr(resp, "success", False)), "error": getattr(resp, "error", "")}

    def delete_device(self, jid: str) -> Dict[str, Any]:
        if not self.stub:
            return {"success": False, "error": "stub_not_ready"}
        resp = self.stub.DeleteDevice(whatsapp_pb2.DeviceID(jid=jid))
        return {"success": bool(getattr(resp, "success", False)), "error": getattr(resp, "error", "")}


whatsapp_transport = WhatsAppTransport()
