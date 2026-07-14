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
        return self._clients.pop(_norm_key(identifier), None) is not None

    def clear(self) -> int:
        n = len(self._clients)
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


sim_client_cache = SimClientCache()