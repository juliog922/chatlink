# src/chatlink_bot/logic/sim_clients.py
"""
Simulation client cache — an in-memory layer of test clients, checked BEFORE
the SAGE database in every client lookup.

Purpose: register phone numbers / email addresses for testing (a second
personal phone, a colleague's mailbox) without ever writing to SAGE. Entries
are CRUD-managed through /api/test/clients, live only in this process's
memory (a restart clears them — that is the point: nothing persists), and are
treated downstream exactly like real SAGE clients: they pass the gatekeeper,
their name feeds the prompt, their code lands in the order xlsx.

Lookup order everywhere: 1) this cache  2) SAGE. In production the cache is
empty, so the cost is one dict lookup and behavior is unchanged.
"""
import re
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

_re_non_digit = re.compile(r"\D+")


def _norm_key(identifier: str) -> str:
    """Phones: digits only, no country-code fuss beyond stripping symbols.
    Emails: lowercase. '+34 600-111-222' and '34600111222' are the same key."""
    ident = (identifier or "").strip()
    if "@" in ident:
        return ident.lower()
    digits = _re_non_digit.sub("", ident)
    return digits or ident.lower()


class SimClientCache:
    def __init__(self) -> None:
        self._clients: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------- CRUD
    def upsert(self, identifier: str, name: Optional[str] = None,
               code: Optional[str] = None, notes: str = "") -> Dict[str, Any]:
        key = _norm_key(identifier)
        if not key:
            raise ValueError("identifier vacío")
        entry = self._clients.get(key, {})
        entry.update({
            "identifier": identifier.strip(),
            "key": key,
            "name": (name or entry.get("name") or f"Cliente Test {identifier.strip()}").strip(),
            "code": (code or entry.get("code") or f"TEST-{key[-6:] or key}").strip(),
            "notes": notes or entry.get("notes", ""),
            "created_at": entry.get("created_at") or time.time(),
            "updated_at": time.time(),
        })
        self._clients[key] = entry
        return dict(entry)

    def get(self, identifier: str) -> Optional[Dict[str, Any]]:
        return dict(self._clients[k]) if (k := _norm_key(identifier)) in self._clients else None

    def delete(self, identifier: str) -> bool:
        entry = self._clients.pop(_norm_key(identifier), None)
        if entry is None:
            return False
        # A test client's fake order history dies with it, or it would linger
        # under a CodigoCliente nobody can resolve any more.
        sim_history_cache.delete(entry.get("code", ""))
        return True

    def clear(self) -> int:
        n = len(self._clients)
        for entry in self._clients.values():
            sim_history_cache.delete(entry.get("code", ""))
        self._clients.clear()
        return n

    def list(self) -> List[Dict[str, Any]]:
        return [dict(e) for e in sorted(self._clients.values(), key=lambda e: e["created_at"])]

    # ------------------------------------------------- SAGE-compatible view
    def resolve(self, identifier: str) -> Optional[SimpleNamespace]:
        """SAGE-row-shaped object (CodigoCliente / Nombre / RazonSocial) so
        handlers treat cache hits exactly like database clients."""
        entry = self.get(identifier)
        if not entry:
            return None
        return SimpleNamespace(
            CodigoCliente=entry["code"],
            Nombre=entry["name"],
            RazonSocial=entry["name"],
            Telefono=entry["identifier"] if "@" not in entry["identifier"] else "",
            Email=entry["identifier"] if "@" in entry["identifier"] else "",
            _sim_client=True,
        )


class SimHistoryCache:
    """Fake ORDER HISTORY for test clients, keyed by CodigoCliente.

    Why this exists
    ---------------
    A test client resolves to CodigoCliente "TEST-600222", which does not exist
    in SAGE, so its order history comes back empty and the history-resolution
    layer can never be exercised without a real recurring client. This cache
    lets a test client carry a plausible purchase history built from REAL
    catalog products (real codes, real descriptions, real brands — sampled from
    the indexed catalog), with fabricated order statistics.

    It is checked BEFORE Postgres and SAGE inside ClientHistoryRetriever.load(),
    mirroring the lookup order the client resolution already uses: 1) sim cache,
    2) the real source. In production it is empty, so the cost is one dict
    lookup and nothing changes.

    Memory only. A restart clears it — that is the point: a simulation must
    never leave rows behind in a database.

    Rows are plain dicts so this module stays stdlib-only (importing Qdrant or
    the ORM here would risk an import cycle with ai/rag.py). The SHAPE is what
    ProductRow expects; ai/rag.py does the conversion.
    """

    def __init__(self) -> None:
        self._orders: Dict[str, List[Dict[str, Any]]] = {}

    def set(self, codigo_cliente: str, products: List[Dict[str, Any]]) -> int:
        code = (codigo_cliente or "").strip()
        if not code:
            raise ValueError("codigo_cliente vacío")
        self._orders[code] = list(products)
        return len(self._orders[code])

    def extend(self, codigo_cliente: str, products: List[Dict[str, Any]]) -> int:
        """Add products, keeping one row per article code (last write wins)."""
        code = (codigo_cliente or "").strip()
        if not code:
            raise ValueError("codigo_cliente vacío")
        existing = {p["codigo"]: p for p in self._orders.get(code, [])}
        for p in products:
            existing[p["codigo"]] = p
        self._orders[code] = list(existing.values())
        return len(self._orders[code])

    def get(self, codigo_cliente: str) -> Optional[List[Dict[str, Any]]]:
        """The fake history, or None when this client has none.

        None and [] mean different things on purpose: None = not a simulated
        history, fall through to SAGE; [] = a deliberately EMPTY simulated
        history, which is a valid test (the "client with no purchases" path).
        """
        rows = self._orders.get((codigo_cliente or "").strip())
        return [dict(r) for r in rows] if rows is not None else None

    def has(self, codigo_cliente: str) -> bool:
        return (codigo_cliente or "").strip() in self._orders

    def count(self, codigo_cliente: str) -> int:
        return len(self._orders.get((codigo_cliente or "").strip()) or [])

    def delete(self, codigo_cliente: str) -> bool:
        return self._orders.pop((codigo_cliente or "").strip(), None) is not None

    def clear(self) -> int:
        n = len(self._orders)
        self._orders.clear()
        return n


sim_history_cache = SimHistoryCache()


sim_client_cache = SimClientCache()