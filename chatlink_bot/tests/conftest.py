"""Scenario harness: the real app against a real Postgres; fakes only at the
edges (WhatsApp, email, SAGE, the clock, and Tachyon's two calls).

    createdb kapa_test
    POSTGRES_HOST=localhost POSTGRES_USER=postgres POSTGRES_PASSWORD=postgres \
        pytest chatlink_bot/tests -q

Each test reads like a scenario row: the client or salesman says something
at a time, the loop ticks, and the test checks what was sent and stored.
"""
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from types import SimpleNamespace as NS

import pytest

os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_USER", "postgres")
os.environ.setdefault("POSTGRES_PASSWORD", "postgres")
os.environ.setdefault("POSTGRES_DB", "kapa_test")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ["RESPONSE_DELAY_MINUTES"] = "15"
os.environ["ALERT_EMAILS"] = ""
os.environ["HEARTBEAT_URL"] = ""
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sqlalchemy import select  # noqa: E402

from chatlink_bot import database, handlers, models  # noqa: E402
from chatlink_bot.ai import inference, llm  # noqa: E402
from chatlink_bot.ai.rag import ProductRow, SyncStats  # noqa: E402
from chatlink_bot.logic import loop  # noqa: E402

SALESMAN = "34600000001"
CLIENT = "34611111111"
DAY = datetime(2026, 10, 22, tzinfo=timezone.utc)


def product(code, name, brand=""):
    return ProductRow(codigo=code, nombre=name, marca=brand, veces=3)


def pedir(texto, qty=None, evidencia=None):
    a = {"tipo": "pedir", "producto_texto": texto, "evidencia": evidencia or texto,
         "cantidad_modo": "total" if qty else "sin_cambio"}
    if qty:
        a["cantidad_valor"] = qty
    return a


def act(tipo, evidencia="", **kw):
    return {"tipo": tipo, "evidencia": evidencia, **kw}


class Kapa:
    def __init__(self, monkeypatch):
        self.loop = asyncio.new_event_loop()
        self.now = DAY.replace(hour=9)
        self.sent, self.emails, self.alerts = [], [], []
        self.extractions, self.laya = [], []
        self.clients = {CLIENT: ("000139", "Perfumería Luz", True)}
        self.history = {"000139": []}
        self.send_ok = True
        self.email_ok = True
        self.inference_up = True
        self.laya_calls = 0
        mp = monkeypatch

        mp.setattr(handlers, "_now_utc", lambda: self.now)
        mp.setattr(handlers.whatsapp_transport, "send_message", self._wa_send)
        mp.setattr(handlers.email_transport, "send_email", self._email)
        mp.setattr(handlers.email_transport, "send_email_as", self._email_as)
        for mod in (handlers, loop):
            mp.setattr(mod, "alert", lambda key, text, also_to=(): self.alerts.append((key, text)) or True)
        mp.setattr(handlers, "resolve_client_by_phone", self._identity)
        mp.setattr(handlers, "resolve_client_identity", self._identity)
        mp.setattr(handlers.client_history, "load", self._history)

        async def no_codes(codes):
            return {c: None for c in codes}
        mp.setattr(handlers, "_sage_code_lookup", no_codes)
        mp.setattr(inference, "chat", self._chat)
        mp.setattr(inference, "decide", self._decide)

        self.run(self._reset_db())

    # ------------------------------------------------------------ fakes
    def _wa_send(self, to_phone, text, from_jid=None, **_):
        if not self.send_ok:
            return {"success": False, "error": "device not connected"}
        self.sent.append(NS(to=to_phone, text=text, from_jid=from_jid))
        return {"success": True, "error": ""}

    def _email(self, to_email, subject, body, attachments=None):
        if not self.email_ok:
            return False, "smtp down"
        self.emails.append(NS(to=to_email, subject=subject, body=body, attachments=attachments or []))
        return True, None

    def _email_as(self, from_email, to_email, subject, body, attachments=None):
        self.sent.append(NS(to=to_email, text=body, from_jid=from_email))
        return True, None

    async def _identity(self, ident):
        hit = self.clients.get(ident)
        if not hit:
            return handlers.ClientIdentity()
        code, name, unique = hit
        return handlers.ClientIdentity(client=NS(Nombre=name, CodigoCliente=code), code=code if unique else None,
                                       unique=unique, source="sage", match_kind="suffix",
                                       candidate_codes=[code] if unique else [code, "999999"])

    async def _history(self, code, force=False):
        rows = self.history.get(code, [])
        return rows, SyncStats(codigo_cliente=code, status="fresh", n_products=len(rows))

    def _chat(self, prompt, images=(), timeout=None):
        if not self.inference_up:
            raise inference.InferenceError("tachyon down")
        if self.extractions:
            return json.dumps(self.extractions.pop(0), ensure_ascii=False)
        return json.dumps({"hablar": True, "actos": []})

    def _decide(self, state, questions, timeout=30):
        self.laya_calls += 1
        out = {}
        picks = self.laya.pop(0) if self.laya else {}
        for name, q in questions.items():
            labels = list(q["criteria"])
            pick = picks.get(name)
            if pick is None:   # Laya unsure: spread evenly
                probs = {lab: round(1 / len(labels), 4) for lab in labels}
                out[name] = {"choice": labels[0], "probabilities": probs}
            else:
                lab, p = pick
                probs = {x: (p if x == lab else round((1 - p) / (len(labels) - 1), 4)) for x in labels}
                out[name] = {"choice": lab, "probabilities": probs}
        return out

    # ------------------------------------------------------------ driving
    def run(self, coro):
        return self.loop.run_until_complete(coro)

    async def _reset_db(self):
        async with database.pg_engine.begin() as conn:
            await conn.run_sync(database.PGBase.metadata.drop_all)
        await database.init_schema()
        async with database.AsyncSessionPG() as db:
            db.add(models.User(name="Ana", email="ana@kapa.test", phone=SALESMAN, enabled=True))
            await db.commit()

    def at(self, hhmm, day=0):
        h, m = (int(x) for x in hhmm.split(":"))
        self.now = DAY.replace(hour=h, minute=m) + __import__("datetime").timedelta(days=day)

    def _wa(self, frm, to, text, binary=b"", filename=""):
        msg = NS(from_phone=frm, to_phone=to, text=text, filename=filename, binary=binary,
                 from_jid=f"{frm}:1@s.whatsapp.net", to_jid=f"{to}:1@s.whatsapp.net")
        self.run(handlers.handle_new_message({"normalized": msg}))

    def client_says(self, text, at=None, phone=CLIENT, understood=None):
        """`understood`: what Gemma extracts from this message (hablar, actos)."""
        if at:
            self.at(at)
        if understood is not None:
            self.extractions.append(understood)
        self._wa(phone, SALESMAN, text)

    def salesman_says(self, text, at=None, to=CLIENT):
        if at:
            self.at(at)
        self._wa(SALESMAN, to, text)

    def tick(self, at=None):
        if at:
            self.at(at)
        self.run(loop.tick(now=self.now))

    def replies(self, to=CLIENT):
        return [s.text for s in self.sent if s.to == to]

    def last_reply(self, to=CLIENT):
        r = self.replies(to)
        return r[-1] if r else None

    def session(self, client=CLIENT):
        async def get():
            async with database.AsyncSessionPG() as db:
                return (await db.execute(select(models.ConversationSession).where(
                    models.ConversationSession.client_id == client))).scalars().first()
        return self.run(get())

    def cart(self, client=CLIENT):
        s = self.session(client)
        return [(i["code"], i["qty"]) for i in (s.cart if s else [])]

    def rows(self, model):
        async def get():
            async with database.AsyncSessionPG() as db:
                return (await db.execute(select(model).order_by(*model.__table__.primary_key.columns))).scalars().all()
        return self.run(get())

    def close(self):
        self.run(database.pg_engine.dispose())
        self.loop.close()


@pytest.fixture
def kapa(monkeypatch):
    k = Kapa(monkeypatch)
    yield k
    k.close()
