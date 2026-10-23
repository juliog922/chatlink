# chatlink_bot/src/chatlink_bot/ai/llm.py
"""
Kapa v3 — tubería tipada por turno.

    Etapa 0  código   preproceso estructural            (§ PREPROCESO)
    Etapa 1  Gemma    extracción -> actos               (§ ACTOS + § TURNO)
    Etapa 2  código   resolución de referencias         (§ RESOLUCION)
             Laya     elige entre candidatos, solo ambiguos
    Etapa 3  código   reductor: ÚNICO escritor          (§ REDUCTOR)
    Etapa 4  código   la plantilla ES la respuesta      (§ RENDER)

El modelo nunca ejecuta, nunca redacta el mensaje y nunca emite códigos: solo
spans del mensaje, enums e índices. El único escritor de carrito/open_items
es `aplicar()`; el mensaje se construye desde los hechos y el carrito, así
que no puede contradecirlos. Inferencia: ai/inference.py (Tachyon).
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from . import inference
from .rag import coverage_rank, tokenize, variants

logger = logging.getLogger("LLM")


# ============================================================================
# ACTOS — esquema, parseo y validación mecánica (invariante 4)
# ============================================================================

TIPOS = ("pedir", "modificar", "elegir", "rechazar", "confirmar",
         "cerrar", "repetir_ultimo", "cantidad_global", "otro")
MODOS = ("total", "incremento", "decremento", "eliminar", "sin_cambio")
SUBTIPOS_OTRO = ("saludo", "despedida", "agradecimiento", "estado_pedido",
                 "pide_comercial", "fuera_ambito",
                 "precio_stock_factura", "especificacion", "queja",
                 "escalada", "rechazo_bot", "pregunta_bot")

# Numerales cuya presencia en la evidencia de un `eliminar` lo degrada a
# pregunta ("¿quito la línea o resto N?"). Clase cerrada: dígitos + cardinales
# bajos. Es estructura, no vocabulario semántico.
_RE_NUMERAL = re.compile(
    r"\d+|\b(un[oa]?|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)\b", re.I)

_RE_DIGITS = re.compile(r"\d+")

# Numerales en palabra: cuentan como evidencia de su valor (invariante 4).
# Clase cerrada y estructural — no es "detección de intención por vocabulario",
# es la tabla de conversión palabra->número del idioma.
_PALABRA_NUM = {"un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4,
                "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
                "diez": 10, "once": 11, "doce": 12, "docena": 12, "trece": 13,
                "catorce": 14, "quince": 15, "veinte": 20}


def _numeros_en(texto: str) -> set:
    """Todos los números presentes en un texto: dígitos + numerales en palabra
    ('media docena' -> 6)."""
    t = (texto or "").lower()
    nums = {int(d) for d in _RE_DIGITS.findall(t)}
    palabras = re.findall(r"[a-záéíóúñ]+", t)
    for i, w in enumerate(palabras):
        if w in _PALABRA_NUM:
            v = _PALABRA_NUM[w]
            if w == "docena" and i > 0 and palabras[i - 1] == "media":
                v = 6
            nums.add(v)
    return nums


@dataclass
class Cantidad:
    modo: str = "sin_cambio"           # MODOS
    valor: Optional[int] = None        # obligatorio salvo sin_cambio/eliminar


@dataclass
class Acto:
    tipo: str
    evidencia: str = ""                # span literal del mensaje
    producto_texto: str = ""           # pedir
    objetivo_texto: str = ""           # modificar / elegir / rechazar
    cantidad: Cantidad = field(default_factory=Cantidad)
    valor: Optional[int] = None        # cantidad_global
    subtipo: str = ""                  # otro
    # marcas que añade la validación (no vienen del modelo):
    eliminar_con_numeral: bool = False

    def as_log(self) -> str:
        c = f"{self.cantidad.modo}:{self.cantidad.valor}" if self.tipo in ("pedir", "modificar") else ""
        return (f"{self.tipo}({self.producto_texto or self.objetivo_texto or self.subtipo or self.valor}"
                + (f" {c}" if c else "") + f" | ev={self.evidencia[:40]!r})")


# ---------------------------------------------------------------- JSON schema
# Plano a propósito: los motores estilo Ollama restringen bien objetos planos
# con enums; oneOf/anyOf por tipo no es fiable. La coherencia por-tipo la
# valida `validar_actos`.
ACTOS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "actos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tipo": {"type": "string", "enum": list(TIPOS)},
                    "evidencia": {"type": "string"},
                    "producto_texto": {"type": "string"},
                    "objetivo_texto": {"type": "string"},
                    "cantidad_modo": {"type": "string", "enum": list(MODOS)},
                    "cantidad_valor": {"type": ["integer", "null"]},
                    "valor": {"type": ["integer", "null"]},
                    "subtipo": {"type": "string", "enum": list(SUBTIPOS_OTRO) + [""]},
                },
                "required": ["tipo", "evidencia"],
            },
        }
    },
    "required": ["actos"],
}

# Schema del match por lotes (resolucion.py): salida por índice, jamás código.
MATCH_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": "integer"},
                    "veredicto": {"type": "string", "enum": ["unico", "varios", "ninguno"]},
                    "indice": {"type": ["integer", "null"]},
                },
                "required": ["item", "veredicto"],
            },
        }
    },
    "required": ["items"],
}


# ------------------------------------------------------------------ parseo
def _to_int(v: Any) -> Optional[int]:
    try:
        if v is None or isinstance(v, bool):
            return None
        n = int(v)
        return n if 0 <= n <= 100000 else None
    except (TypeError, ValueError):
        return None


def parse_actos(data: Any) -> List[Acto]:
    """Dict (ya json.loads) -> [Acto]. Tolerante en tipos, estricto en enums."""
    out: List[Acto] = []
    bruto = data.get("actos") if isinstance(data, dict) else None
    if not isinstance(bruto, list):     # el 2B a veces devuelve int/str/dict aquí
        bruto = []
    for raw in bruto:
        if not isinstance(raw, dict):
            continue
        tipo = str(raw.get("tipo") or "").strip()
        if tipo not in TIPOS:
            continue
        out.append(Acto(
            tipo=tipo,
            evidencia=str(raw.get("evidencia") or "").strip(),
            producto_texto=str(raw.get("producto_texto") or "").strip(),
            objetivo_texto=str(raw.get("objetivo_texto") or "").strip(),
            cantidad=Cantidad(
                modo=(raw.get("cantidad_modo") if raw.get("cantidad_modo") in MODOS
                      else "sin_cambio"),
                valor=_to_int(raw.get("cantidad_valor")),
            ),
            valor=_to_int(raw.get("valor")),
            subtipo=(raw.get("subtipo") if raw.get("subtipo") in SUBTIPOS_OTRO else ""),
        ))
    return out


# -------------------------------------------------------------- validación
def validar_actos(actos: List[Acto], mensaje: str) -> List[Acto]:
    """Filtro mecánico post-schema. Deja pasar solo actos coherentes:

    - invariante 4: un `valor` de cantidad debe aparecer como dígitos en la
      EVIDENCIA (o en el mensaje si la evidencia vino recortada). Si no, la
      cantidad se degrada a sin_cambio/None: nunca entra un número inventado.
    - la evidencia debe existir en el mensaje (subcadena, espacios laxos);
      si no, se conserva el acto pero la evidencia se vacía (no bloquea:
      bloquear aquí silenciaría pedidos, inv. 7).
    - `eliminar` con numeral en la evidencia -> se marca para que el reductor
      pregunte en vez de borrar (caso 7 del catálogo).
    - coherencia por tipo: pedir sin producto_texto se descarta; modificar/
      elegir/rechazar sin objetivo_texto se descartan (nada que ligar).
    """
    msg_flat = re.sub(r"\s+", " ", mensaje or "").lower()
    ok: List[Acto] = []
    for a in actos:
        ev_flat = re.sub(r"\s+", " ", a.evidencia).lower()
        if ev_flat and ev_flat not in msg_flat:
            a.evidencia = ""
            ev_flat = ""

        base = ev_flat or msg_flat

        # invariante 4 por string: el valor debe estar en la evidencia
        # (dígitos o numeral en palabra); si no, jamás entra un número.
        if a.cantidad.valor is not None and a.cantidad.valor not in _numeros_en(base):
            a.cantidad = Cantidad(modo="sin_cambio", valor=None)
        if a.tipo == "cantidad_global" and a.valor is not None \
                and a.valor not in _numeros_en(base):
            a.valor = None

        if a.cantidad.modo in ("total", "incremento", "decremento") and a.cantidad.valor is None:
            a.cantidad = Cantidad(modo="sin_cambio", valor=None)

        if a.tipo == "modificar" and a.cantidad.modo == "eliminar" \
                and _RE_NUMERAL.search(a.evidencia or ""):
            a.eliminar_con_numeral = True

        # Sin evidencia = el modelo no lo sacó del mensaje (típico: repite la
        # lista que la Etapa 0 ya consumió, o rellena de memoria). Un acto que
        # TOCA el pedido sin fundamento en el texto no entra. `otro`,
        # `confirmar`, `cerrar` y `repetir_ultimo` se libran: pueden nacer de
        # un mensaje muy corto ("sí", "vale") cuya evidencia es trivial.
        if not a.evidencia and a.tipo in ("pedir", "modificar", "elegir",
                                          "rechazar", "cantidad_global"):
            continue
        if a.tipo == "pedir" and not a.producto_texto:
            continue
        if a.tipo in ("modificar", "elegir", "rechazar") and not a.objetivo_texto:
            if a.tipo == "rechazar":
                a.objetivo_texto = ""   # rechazo global de la última pregunta: permitido
            else:
                continue
        if a.tipo == "cantidad_global" and a.valor is None:
            continue
        if a.tipo == "otro" and not a.subtipo:
            a.subtipo = "fuera_ambito"
        ok.append(a)
    return ok

# ============================================================================
# PREPROCESO — Etapa 0, estructural puro
# ============================================================================

# Marcadores que inyecta handlers._MEDIA_PREFIX (+ el genérico de documentos).
_MEDIA_MARKERS = ("[Texto en Imagen]:", "[Audio transcrito]:", "[Documento]:")

# Marcas de repetición al inicio de línea: "" "" | " " | «» | ídem | id.
_RE_DITTO = re.compile(r'^\s*[-•*\u2022]?\s*((?:["\u201c\u201d\u00ab\u00bb]{1,2}\s*)+|idem\b\.?|ídem\b\.?|id\.\s)',
                       re.I)
_RE_DITTO_MARK = re.compile(r'["\u201c\u201d\u00ab\u00bb]{1,2}|idem\.?|ídem\.?|id\.', re.I)

_RE_BULLET = re.compile(r'^\s*[-•*\u2022]\s*')

# "6 de cada uno" / "de cada una 6" / "todas a 6" / "6 de cada" / "6 de todos"
_RE_GLOBAL_QTY = re.compile(
    r'^\s*(?:'
    r'(\d{1,4})\s*(?:uds?\.?|unidades?)?\s*de\s+cada(?:\s+un[oa]s?)?'
    r'|de\s+cada\s+un[oa]?\s+(\d{1,4})'
    r'|tod[oa]s\s+a\s+(\d{1,4})'
    r'|(\d{1,4})\s+de\s+tod[oa]s'
    r')\s*\.?\s*$', re.I)

_RE_CODE_SHAPE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9\-\./]{2,20}$')

# Línea de pedido con cantidad EXPLÍCITA al final: ", 12" o "12 uds" (la
# unidad o la coma son obligatorias — "KAJAL 401" es nombre, no 401 uds).
_RE_TRAIL_QTY = re.compile(
    r'^(?P<name>.+?)(?:'
    r'[,;]\s*(?P<q1>\d{1,4})\s*(?:uds?|unds?|unidades|u)?\.?'      # "..., 10" / "..., 10u"
    r'|\s+x?\s*(?P<q2>\d{1,4})\s*(?:uds?|unds?|unidades|u)\.?'     # "... 10u" / "... x10 uds"
    r')\s*$', re.I)


def _parse_linea_pedido(line: str) -> Tuple[str, Optional[int]]:
    """(nombre, cantidad|None) de una línea de lista. Estructura pura."""
    body = _RE_BULLET.sub("", line).strip()
    m = _RE_TRAIL_QTY.match(body)
    if m:
        return m.group("name").strip(" .,;-"), int(m.group("q1") or m.group("q2"))
    return body.strip(" .,;-"), None


def code_shaped(token: str) -> bool:
    """Token compacto con al menos un dígito y sin espacios: forma de código."""
    t = (token or "").strip()
    return bool(t) and bool(_RE_CODE_SHAPE.match(t)) and any(c.isdigit() for c in t) \
        and not t.isdigit() or (t.isdigit() and len(t) >= 5)


def codes_in_text(text: str) -> List[str]:
    return [t for t in re.split(r"[\s,;:()]+", text or "") if t and code_shaped(t)]


@dataclass
class PreResult:
    texto: str                       # resto del mensaje (lo que SÍ es lenguaje)
    lineas: List[str] = field(default_factory=list)
    # líneas de lista consumidas estructuralmente: (nombre, qty|None, línea):
    pedidos: List[Tuple[str, Optional[int], str]] = field(default_factory=list)
    fuente_media: bool = False
    cantidad_global: Optional[int] = None   # solo formas canónicas (regex)
    log: List[str] = field(default_factory=list)


def _expand_ditto_line(line: str, prev_expanded: str) -> Tuple[str, bool]:
    """k marcas al inicio -> hereda los primeros k tokens de la línea anterior
    expandida. El resto de la línea se conserva tal cual."""
    m = _RE_DITTO.match(line)
    if not m or not prev_expanded:
        return line, False
    marks = _RE_DITTO_MARK.findall(m.group(1))
    k = max(1, len(marks))
    prev_name, _ = _parse_linea_pedido(prev_expanded)   # sin su cantidad final
    name_tokens = prev_name.split()
    inherited = name_tokens[:k]
    rest = line[m.end():].strip()
    bullet = _RE_BULLET.match(line)
    prefix = bullet.group(0) if bullet else ""
    return (prefix + " ".join(inherited) + (" " + rest if rest else "")).rstrip(), True


def preprocesar(mensaje: str) -> PreResult:
    fuente_media = False
    text = mensaje or ""
    for mk in _MEDIA_MARKERS:
        if mk in text:
            fuente_media = True
            text = text.replace(mk, " ")
    raw_lines = [ln.rstrip() for ln in text.splitlines()]

    expanded: List[str] = []
    n_ditto = 0
    prev = ""
    for ln in raw_lines:
        if not ln.strip():
            expanded.append(ln)
            continue
        new_ln, did = _expand_ditto_line(ln, prev)
        if did:
            n_ditto += 1
        expanded.append(new_ln)
        prev = new_ln if new_ln.strip() else prev

    cantidad_global: Optional[int] = None
    kept: List[str] = []
    pedidos: List[Tuple[str, Optional[int], str]] = []
    for ln in expanded:
        m = _RE_GLOBAL_QTY.match(ln)
        if m:
            val = next((g for g in m.groups() if g), None)
            if val:
                cantidad_global = int(val)
                continue  # la línea se consume: es una instrucción, no un producto
        # Una línea con viñeta es un renglón de pedido: estructura, no
        # lenguaje. Se consume aquí y NO pasa por la extracción — el 2B ha
        # demostrado perder renglones al enumerar listas (log 18:21/18:22).
        if _RE_BULLET.match(ln):
            nombre, qty = _parse_linea_pedido(ln)
            if nombre:
                pedidos.append((nombre, qty, ln.strip()))
                continue
        kept.append(ln)

    res = PreResult(
        texto="\n".join(kept).strip(),
        lineas=[ln for ln in kept if ln.strip()],
        pedidos=pedidos,
        fuente_media=fuente_media,
        cantidad_global=cantidad_global,
    )
    if pedidos:
        res.log.append(f"lineas_pedido={len(pedidos)}")
    if n_ditto:
        res.log.append(f"ditto x{n_ditto}")
    if cantidad_global is not None:
        res.log.append(f"cantidad_global={cantidad_global} (canónica)")
    if fuente_media:
        res.log.append("fuente_media")
    return res

# ============================================================================
# REDUCTOR — Etapa 3, único escritor de estado
# ============================================================================

SI_NO_TIPOS = ("propuesta_media", "propuesta_repetir", "propuesta_cierre", "propuesta_baja")

# Un producto que sigue sin resolverse tras este nº de turnos con el cliente
# aportando datos SIN progreso -> se deja al comercial (no forzar el pedido).
MAX_INTENTOS_PRODUCTO = int(os.getenv("PIPE_MAX_INTENTOS_PRODUCTO", "2"))


# ------------------------------------------------------------------- estado
def _coerce_open_item(it: Any) -> Optional[Dict[str, Any]]:
    """Tolera formatos antiguos de open_items (estado='enriquecer', claves
    pedido/consulta) convirtiéndolos al contenedor tipado."""
    if not isinstance(it, dict):
        return None
    if it.get("tipo") in ("producto",) + SI_NO_TIPOS + ("conflicto_cantidad",):
        return dict(it)
    texto = str(it.get("texto") or it.get("pedido") or it.get("consulta") or "").strip()
    if not texto:
        return None
    ops = []
    for o in it.get("opciones") or []:
        if isinstance(o, dict):
            ops.append({"code": str(o.get("code") or o.get("codigo") or ""),
                        "name": str(o.get("name") or o.get("nombre") or ""),
                        "brand": str(o.get("brand") or o.get("marca") or ""),
                        "distintivo": str(o.get("distintivo") or "")})
    qty = it.get("qty") if isinstance(it.get("qty"), int) else (
        it.get("cantidad") if isinstance(it.get("cantidad"), int) else None)
    return {"tipo": "producto", "texto": texto, "qty": qty,
            "opciones": ops, "rechazados": list(it.get("rechazados") or [])}


@dataclass
class Estado:
    order_status: str = "IDLE"                      # IDLE | BUILDING | CLOSED
    cart: List[Dict[str, Any]] = field(default_factory=list)   # {code, qty, name?, fuente?}
    open_items: List[Dict[str, Any]] = field(default_factory=list)
    last_closed_cart: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_session(cls, s: Dict[str, Any]) -> "Estado":
        items = [ci for it in (s.get("open_items") or []) if (ci := _coerce_open_item(it))]
        cart = []
        for c in (s.get("cart") or []):
            if isinstance(c, dict) and c.get("code"):
                cart.append({"code": str(c["code"]), "qty": int(c.get("qty") or 1),
                             "name": str(c.get("name") or ""),
                             "fuente": str(c.get("fuente") or "")})
        return cls(order_status=str(s.get("order_status") or "IDLE"),
                   cart=cart, open_items=items,
                   last_closed_cart=list(s.get("last_closed_cart") or []))

    # helpers --------------------------------------------------------------
    def linea(self, code: str) -> Optional[Dict[str, Any]]:
        return next((l for l in self.cart if l["code"] == code), None)

    def si_no_vivo(self) -> Optional[Dict[str, Any]]:
        return next((i for i in self.open_items if i.get("tipo") in SI_NO_TIPOS), None)

    def productos_abiertos(self) -> List[Dict[str, Any]]:
        return [i for i in self.open_items if i.get("tipo") == "producto"]


# -------------------------------------------------------------------- hechos
@dataclass
class Hecho:
    """Unidad tipada de 'qué se dirá'. El renderizador serializa; nada más."""
    tipo: str
    datos: Dict[str, Any] = field(default_factory=dict)

    # tipos:
    # confirmado {code,name,qty}            eliminado {code,name}
    # actualizado {code,name,qty}           sin_cambio_confirmado {code,name,qty}
    # pregunta_opciones {texto,qty,opciones,indistinguibles}
    # conflicto_cantidad {code,name,opciones_qty}
    # pregunta_baja {code,name}             pregunta_eliminar_o_restar {code,name,n}
    # sin_candidatos {texto}                codigo_no_existe {code}
    # rechazo_registrado {texto}            referencia_ambigua {texto,candidatas}
    # propuesta {tipo,items,titulo}         cerrado {cart}
    # derivacion {motivo}                   opt_out {}   escalada {}
    # cortesia {subtipo}                    sin_entender {}
    # recordatorio_pendiente {item}         guia {}


@dataclass
class ResultadoTurno:
    estado: Estado
    hechos: List[Hecho] = field(default_factory=list)
    descartes: List[str] = field(default_factory=list)   # log de actos vetados

    @property
    def handoff(self) -> bool:
        return any(h.tipo in ("derivacion", "escalada", "aviso_comercial") for h in self.hechos)

    @property
    def opt_out(self) -> bool:
        return any(h.tipo == "opt_out" for h in self.hechos)


# ---------------------------------------------------------------- operaciones
# Producidas por resolucion.py. Shape (dicts para mantener la frontera fina):
#   {"op":"add","code","name","qty":int|None,"fuente":str,"brand":str}
#   {"op":"modificar","code","name","modo","valor":int|None,
#    "eliminar_con_numeral":bool,"valor_numeral":int|None}
#   {"op":"opciones","texto","qty":int|None,"opciones":[opt],"indistinguibles":bool}
#   {"op":"sin_candidatos","texto","qty":int|None}
#   {"op":"codigo_no_existe","code"}
#   {"op":"rechazo","item_texto":str|None}
#   {"op":"elige","item_texto","code","name"}          (opción resuelta)
#   {"op":"elige_qty","code","valor"}                  (conflicto resuelto)
#   {"op":"referencia_ambigua","texto","candidatas":[str]}
#   {"op":"confirmar"} {"op":"cerrar"} {"op":"cantidad_global","valor"}
#   {"op":"repetir_ultimo","qty_override":int|None}
#   {"op":"otro","subtipo"}


def aplicar(estado_in: Estado, ops: List[Dict[str, Any]],
            *, fuente_media: bool = False,
            cantidad_global_pre: Optional[int] = None) -> ResultadoTurno:
    e = copy.deepcopy(estado_in)
    hechos: List[Hecho] = []
    descartes: List[str] = []
    # códigos añadidos/actualizados ESTE turno sin cantidad explícita (regla 11)
    added_no_qty: List[str] = []
    # adds de este turno agrupados para la regla 3
    seen_adds: Dict[str, List[Optional[int]]] = {}
    media_batch: List[Dict[str, Any]] = []

    def _emit(_t: str, **datos: Any) -> None:
        hechos.append(Hecho(tipo=_t, datos=datos))

    def _nueva_si_no(item: Dict[str, Any], titulo: str) -> bool:
        """Regla 10: una sola propuesta sí/no viva."""
        if e.si_no_vivo() is not None:
            descartes.append(f"si_no duplicada vetada: {item.get('tipo')}")
            return False
        e.open_items.append(item)
        _emit("propuesta", tipo=item["tipo"], items=item.get("items") or [], titulo=titulo)
        return True

    def _set_qty(code: str, name: str, modo: str, valor: Optional[int]) -> None:
        ln = e.linea(code)
        if ln is None:
            descartes.append(f"modificar sobre código fuera de carrito: {code}")
            _emit("referencia_ambigua", texto=name or code, candidatas=[])
            return
        actual = int(ln["qty"])
        if modo == "sin_cambio" or valor is None:
            _emit("sin_cambio_confirmado", code=code, name=ln.get("name") or name, qty=actual)
            return
        nuevo = {"total": valor,
                 "incremento": actual + valor,
                 "decremento": actual - valor}.get(modo, actual)
        if nuevo <= 0:
            # D7: no escribe, no borra; pregunta.
            if _nueva_si_no({"tipo": "propuesta_baja", "code": code,
                             "name": ln.get("name") or name, "items": []},
                            titulo=""):
                hechos.pop()  # propuesta genérica fuera; hecho específico:
                _emit("pregunta_baja", code=code, name=ln.get("name") or name)
            return
        ln["qty"] = nuevo
        _emit("actualizado", code=code, name=ln.get("name") or name, qty=nuevo)

    def _add(code: str, name: str, brand: str, qty: Optional[int], fuente: str) -> None:
        seen_adds.setdefault(code, []).append(qty)
        ln = e.linea(code)
        if ln is not None:
            actual = int(ln["qty"])
            if qty is None:
                # ya estaba con cantidad: se conserva (B4.2), y se dice.
                _emit("sin_cambio_confirmado", code=code, name=ln.get("name") or name, qty=actual)
                return
            if qty == actual:
                _emit("sin_cambio_confirmado", code=code, name=ln.get("name") or name, qty=actual)
                return
            # D2: pedir con número sobre código ya en carrito con otra cantidad
            # -> conflicto con números precalculados. Nadie suma salvo el código.
            e.open_items = [i for i in e.open_items
                            if not (i.get("tipo") == "conflicto_cantidad" and i.get("code") == code)]
            opciones_qty = sorted({qty, actual, qty + actual})
            e.open_items.append({"tipo": "conflicto_cantidad", "code": code,
                                 "name": ln.get("name") or name, "opciones_qty": opciones_qty})
            _emit("conflicto_cantidad", code=code, name=ln.get("name") or name,
                  opciones_qty=opciones_qty)
            return
        q = qty if qty is not None else 1     # default 1 SOLO aquí: línea nueva
        e.cart.append({"code": code, "qty": q, "name": name, "fuente": fuente,
                       **({"brand": brand} if brand else {})})
        if qty is None:
            added_no_qty.append(code)
        e.order_status = "BUILDING" if e.order_status != "CLOSED" else e.order_status
        _emit("confirmado", code=code, name=name, qty=q,
              marcado=(fuente == "codigo_cliente"))

    # ---- pase 1: operaciones --------------------------------------------
    for op in ops:
        k = op.get("op")

        if k == "add":
            if fuente_media:
                media_batch.append({"code": op["code"], "name": op.get("name") or "",
                                    "qty": op.get("qty"), "fuente": op.get("fuente") or ""})
                continue
            _add(op["code"], op.get("name") or "", op.get("brand") or "",
                 op.get("qty"), op.get("fuente") or "")

        elif k == "modificar":
            if op.get("modo") == "eliminar":
                ln = e.linea(op["code"])
                if ln is None:
                    descartes.append(f"eliminar sobre código fuera de carrito: {op.get('code')}")
                    continue
                if op.get("eliminar_con_numeral"):
                    _emit("pregunta_eliminar_o_restar", code=ln["code"],
                          name=ln.get("name") or "", n=op.get("valor_numeral") or 1)
                    continue
                e.cart = [l for l in e.cart if l["code"] != ln["code"]]
                _emit("eliminado", code=ln["code"], name=ln.get("name") or "")
            else:
                _set_qty(op["code"], op.get("name") or "", op.get("modo") or "sin_cambio",
                         op.get("valor"))

        elif k == "opciones":
            # (re)abre o actualiza un producto pendiente con sus opciones vivas
            it = next((i for i in e.productos_abiertos()
                       if i.get("texto") == op["texto"]), None)
            if it is None:
                it = {"tipo": "producto", "texto": op["texto"], "qty": op.get("qty"),
                      "opciones": [], "rechazados": [], "_nuevo": True}
                e.open_items.append(it)
            if op.get("qty") is not None:
                it["qty"] = op["qty"]
            it["opciones"] = [o for o in op.get("opciones") or []
                              if o.get("code") not in set(it.get("rechazados") or [])]
            _emit("pregunta_opciones", texto=it["texto"], qty=it.get("qty"),
                  opciones=it["opciones"],
                  indistinguibles=bool(op.get("indistinguibles")))

        elif k == "sin_candidatos":
            it = next((i for i in e.productos_abiertos()
                       if i.get("texto") == op["texto"]), None)
            if it is None:
                e.open_items.append({"tipo": "producto", "texto": op["texto"],
                                     "qty": op.get("qty"), "opciones": [], "rechazados": [],
                                     "_nuevo": True})
            _emit("sin_candidatos", texto=op["texto"])

        elif k == "codigo_no_existe":
            _emit("codigo_no_existe", code=op.get("code") or "")

        elif k == "rechazo":
            objetivo = op.get("item_texto")
            items = e.productos_abiertos()
            it = next((i for i in items if i.get("texto") == objetivo), None) \
                or (items[0] if len(items) == 1 else None)
            if it is None:
                descartes.append("rechazo sin ítem ligable")
                _emit("referencia_ambigua", texto=objetivo or "eso",
                      candidatas=[i["texto"] for i in items])
                continue
            it["rechazados"] = sorted(set(it.get("rechazados") or [])
                                      | {o.get("code") for o in it.get("opciones") or [] if o.get("code")})
            it["opciones"] = []
            _emit("rechazo_registrado", texto=it["texto"])

        elif k == "elige":
            it = next((i for i in e.productos_abiertos()
                       if i.get("texto") == op.get("item_texto")), None)
            qty = (it or {}).get("qty")          # regla 6: hereda la cantidad
            if it is not None:
                e.open_items.remove(it)
            _add(op["code"], op.get("name") or "", op.get("brand") or "", qty,
                 op.get("fuente") or "")

        elif k == "elige_qty":
            it = next((i for i in e.open_items
                       if i.get("tipo") == "conflicto_cantidad" and i.get("code") == op.get("code")), None)
            if it is None or op.get("valor") not in (it.get("opciones_qty") or []):
                descartes.append(f"elige_qty inválido: {op}")
                continue
            e.open_items.remove(it)
            ln = e.linea(op["code"])
            if ln is None:
                e.cart.append({"code": op["code"], "qty": int(op["valor"]),
                               "name": it.get("name") or "", "fuente": ""})
                e.order_status = "BUILDING" if e.order_status != "CLOSED" else e.order_status
                _emit("confirmado", code=op["code"], name=it.get("name") or "",
                      qty=int(op["valor"]), marcado=False)
            else:
                ln["qty"] = int(op["valor"])
                _emit("actualizado", code=op["code"], name=ln.get("name") or "", qty=ln["qty"])

        elif k == "referencia_ambigua":
            _emit("referencia_ambigua", texto=op.get("texto") or "",
                  candidatas=op.get("candidatas") or [])

        elif k == "cantidad_global":
            cantidad_global_pre = op.get("valor") or cantidad_global_pre

        elif k == "repetir_ultimo":
            if not e.last_closed_cart:
                _emit("derivacion", motivo="repetir_sin_historial")
                continue
            qo = op.get("qty_override")
            items = [{"code": c.get("code"), "qty": int(qo or c.get("qty") or 1),
                      "name": c.get("name") or ""} for c in e.last_closed_cart if c.get("code")]
            _nueva_si_no({"tipo": "propuesta_repetir", "items": items}, titulo="repetir")

        elif k == "cerrar":
            if not e.cart:
                _emit("sin_entender")
                continue
            _nueva_si_no({"tipo": "propuesta_cierre",
                          "items": [dict(l) for l in e.cart]}, titulo="cierre")

        elif k == "confirmar":
            vivo = e.si_no_vivo()
            if vivo is None:
                _emit("cortesia", subtipo="confirmacion_sin_pregunta")   # caso 4: jamás escribe
                continue
            e.open_items.remove(vivo)
            t = vivo["tipo"]
            if t == "propuesta_cierre":
                e.order_status = "CLOSED"
                _emit("cerrado", cart=[dict(l) for l in e.cart])
            elif t == "propuesta_baja":
                code = vivo.get("code")
                ln = e.linea(code)
                if ln:
                    e.cart = [l for l in e.cart if l["code"] != code]
                    _emit("eliminado", code=code, name=vivo.get("name") or "")
            else:  # propuesta_media / propuesta_repetir
                for itx in vivo.get("items") or []:
                    _add(itx["code"], itx.get("name") or "", "", int(itx.get("qty") or 1),
                         "media" if t == "propuesta_media" else "repetir")

        elif k == "otro":
            st = op.get("subtipo") or "fuera_ambito"
            if st in ("escalada", "pide_comercial"):
                _emit("aviso_comercial")     # "aviso a X para que se conecte"
            elif st == "rechazo_bot":
                _emit("opt_out")
            elif st in ("precio_stock_factura", "especificacion", "queja", "fuera_ambito"):
                _emit("derivacion", motivo=st)
            elif st == "estado_pedido":
                _emit("recap_pedido")
            elif st in ("saludo", "despedida", "agradecimiento", "pregunta_bot"):
                _emit("cortesia", subtipo=st)

        else:
            descartes.append(f"op desconocida: {k}")

    # ---- pase 2: reglas transversales ------------------------------------
    # media: los add resueltos van a propuesta, no al carrito (B9).
    if media_batch:
        for m in media_batch:
            if m["qty"] is None:
                m["qty"] = 1
        _nueva_si_no({"tipo": "propuesta_media", "items": media_batch}, titulo="media")

    # regla 3 en-mensaje: mismo código, cantidades distintas (caso 3).
    for code, qtys in seen_adds.items():
        distintas = sorted({q for q in qtys if q is not None})
        if len(distintas) > 1:
            ln = e.linea(code)
            opciones_qty = sorted(set(distintas) | {sum(distintas)})
            # la línea escrita queda en suspenso: se retira y se pregunta
            name = (ln or {}).get("name") or code
            e.cart = [l for l in e.cart if l["code"] != code]
            hechos[:] = [h for h in hechos
                         if not (h.tipo in ("confirmado", "actualizado", "sin_cambio_confirmado")
                                 and h.datos.get("code") == code)]
            e.open_items = [i for i in e.open_items
                            if not (i.get("tipo") == "conflicto_cantidad" and i.get("code") == code)]
            e.open_items.append({"tipo": "conflicto_cantidad", "code": code,
                                 "name": name, "opciones_qty": opciones_qty})
            _emit("conflicto_cantidad", code=code, name=name, opciones_qty=opciones_qty)

    # regla 11 (D1): cantidad global solo sobre líneas de ESTE turno sin qty.
    if cantidad_global_pre is not None:
        for code in added_no_qty:
            ln = e.linea(code)
            if ln is not None:
                ln["qty"] = int(cantidad_global_pre)
                _emit("actualizado", code=code, name=ln.get("name") or "", qty=ln["qty"])
        # y sobre ítems abiertos de este turno sin cantidad
        for it in e.productos_abiertos():
            if it.get("qty") is None:
                it["qty"] = int(cantidad_global_pre)

    # ¿hubo progreso en el pedido este turno? Progreso = algo se resolvió
    # (se apuntó/actualizó una línea, se cerró un pendiente, se eligió opción).
    # Mientras el pedido AVANCE, ningún irresoluble suma intentos: la
    # conversación fluye y solo se estanca lo que de verdad no se mueve.
    hubo_progreso = any(h.tipo in ("confirmado", "actualizado", "eliminado",
                                   "conflicto_cantidad", "cerrado", "rechazo_registrado")
                        for h in hechos)

    # regla 7 (B5): toda duda viva emite hecho; sin duplicar los ya emitidos.
    ya = {(h.tipo, h.datos.get("texto") or h.datos.get("code")) for h in hechos}
    for it in e.open_items:
        t = it.get("tipo")
        if t == "producto":
            key = ("pregunta_opciones", it.get("texto"))
            key2 = ("sin_candidatos", it.get("texto"))
            key3 = ("rechazo_registrado", it.get("texto"))
            recien_tratado = key in ya or key2 in ya or key3 in ya
            # Un producto SIN opciones que ya venía de un turno anterior y sigue
            # sin resolverse suma un intento. Agotados los intentos, no se
            # repregunta: se deja al comercial (no forzar el pedido).
            if not it.get("opciones"):
                # sube intento solo si el ítem ya venía de antes Y el turno no
                # avanzó en NADA del pedido. Si hubo progreso, se reinicia: el
                # cliente está aclarando otros y este puede esperar.
                if it.get("_nuevo") or hubo_progreso:
                    it["intentos"] = 0 if hubo_progreso else int(it.get("intentos") or 0)
                else:
                    it["intentos"] = int(it.get("intentos") or 0) + 1
                if it["intentos"] >= MAX_INTENTOS_PRODUCTO:
                    it["_derivar"] = True
                    _emit("derivacion", motivo="producto_no_encontrado", texto=it["texto"])
                    continue
            if recien_tratado:
                continue
            if it.get("opciones"):
                _emit("pregunta_opciones", texto=it["texto"], qty=it.get("qty"),
                      opciones=it["opciones"], indistinguibles=False)
            else:
                _emit("recordatorio_pendiente", item=it["texto"])
        elif t == "conflicto_cantidad" and ("conflicto_cantidad", it.get("code")) not in ya:
            _emit("conflicto_cantidad", code=it["code"], name=it.get("name") or "",
                  opciones_qty=it.get("opciones_qty") or [])
        elif t in SI_NO_TIPOS and not any(h.tipo == "propuesta" and h.datos.get("tipo") == t
                                          for h in hechos) \
                and not any(h.tipo in ("pregunta_baja", "cerrado") for h in hechos):
            if t == "propuesta_baja":
                _emit("pregunta_baja", code=it.get("code"), name=it.get("name") or "")
            else:
                _emit("propuesta", tipo=t, items=it.get("items") or [], titulo="")

    # los productos que agotaron intentos salen del estado (ya derivados)
    e.open_items = [it for it in e.open_items if not it.get("_derivar")]
    for it in e.open_items:
        it.pop("_nuevo", None)   # marca efímera de "creado este turno"

    # invariante 7: un turno con ops y cero hechos es imposible; y con cero
    # ops sobre un mensaje no vacío, resolucion ya emitió sin_entender.
    if ops and not hechos:
        _emit("sin_entender")

    return ResultadoTurno(estado=e, hechos=hechos, descartes=descartes)

# ============================================================================
# RESOLUCION — Etapa 2, ligado de referencias + retrieval + match
# ============================================================================


MATCH_MAX_ITEMS = int(os.getenv("PIPE_MATCH_MAX_ITEMS", "5"))
MAX_OPCIONES = int(os.getenv("PIPE_MAX_OPCIONES", "4"))

Retriever = Callable[[List[str]], Awaitable[Dict[str, List[Dict[str, Any]]]]]
CodeLookup = Callable[[List[str]], Awaitable[Dict[str, Optional[Dict[str, str]]]]]
# match_call(items) -> [{"item":i,"veredicto":...,"indice":...}]  (llm.py lo inyecta)
MatchCall = Callable[[List[Dict[str, Any]]], Awaitable[Optional[List[Dict[str, Any]]]]]


def _canon_tokens(name: str) -> frozenset:
    """Tokens de nombre plegados por el léxico: cada token se representa por
    el mínimo de su clase de variantes, para que CHAMPU y SHAMPOO colisionen
    y los subconjuntos de la dominancia se cumplan donde deben."""
    out = set()
    for t in tokenize(name or ""):
        out.add(min(variants(t)))
    return frozenset(out)


@dataclass
class Ligado:
    """Resultado de ligar una referencia textual contra un espacio cerrado."""
    unico: Optional[Dict[str, Any]] = None
    varios: List[Dict[str, Any]] = field(default_factory=list)


def _ligar(texto: str, espacio: List[Tuple[str, str, Dict[str, Any]]]) -> Ligado:
    """espacio: [(key, texto_matchable, objeto)]. Cobertura total y mínimo de
    extras; empate exacto -> varios (pregunta)."""
    if not espacio:
        return Ligado()
    ranked = coverage_rank(texto, [(k, t, t) for k, t, _ in espacio])
    q_len = len(set(tokenize(texto)))
    if not ranked or q_len == 0:
        return Ligado()
    by_key = {k: obj for k, _, obj in espacio}
    top_cov = ranked[0][1]
    if top_cov == 0:
        return Ligado()
    best = [r for r in ranked if r[1] == top_cov]
    min_extra = min(r[2] for r in best)
    best = [r for r in best if r[2] == min_extra]
    if len(best) == 1:
        return Ligado(unico=by_key[best[0][0]])
    return Ligado(varios=[by_key[r[0]] for r in best])


def _hit_opt(h: Dict[str, Any]) -> Dict[str, str]:
    # `fecha` = lo más nuevo que traiga el recuperador para desempatar gemelos
    # de nombre (FechaAlta del artículo, o la última compra). Cadena ISO
    # comparable, o "" si el retriever aún no la incluye.
    fecha = h.get("FechaAlta") or ""   # campo canónico para "el más nuevo"
    return {"code": str(h.get("CodigoArticulo") or ""),
            "name": str(h.get("DescripcionArticulo") or ""),
            "brand": str(h.get("MarcaProducto") or ""),
            "fuente": str(h.get("fuente") or "pedidos"),
            "fecha": str(fecha),
            "distintivo": ""}


def _norm_nombre(name: str) -> str:
    """Normalización mínima (acentos/mayúsculas/espacios) para detectar nombres
    LITERALMENTE iguales. No intenta juzgar 'casi iguales': eso es semántico y
    lo decide el match LLM, no una regla."""
    import unicodedata
    t = unicodedata.normalize("NFKD", (name or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.split())


def _colapsar_gemelos(cands: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Colapsa SOLO códigos con nombre idéntico (misma cadena): son el mismo
    producto reinscrito tras cambio de EAN. Se queda el más nuevo por `fecha`,
    sin preguntar. Nombres que difieren aunque sea poco NO se tocan aquí — si
    son variantes reales o el mismo producto es juicio del match LLM."""
    grupos: Dict[str, List[Dict[str, str]]] = {}
    orden: List[str] = []
    for c in cands:
        k = _norm_nombre(c["name"])
        if k not in grupos:
            grupos[k] = []; orden.append(k)
        grupos[k].append(c)
    out = []
    for k in orden:
        g = grupos[k]
        out.append(max(g, key=lambda c: c.get("fecha") or "") if any(c.get("fecha") for c in g) else g[0])
    return out


def _es_tester(c: Dict[str, str]) -> bool:
    """Tester / muestra / PLV: no es producto vendible normal. Se reconoce por
    el nombre ('TESTER'/'TEST') o el código acabado en 'T'. El cliente que pide
    'stay matte 402' nunca quiere el tester."""
    name = (c.get("name") or "").upper()
    code = (c.get("code") or "").upper()
    return "TESTER" in name or " TEST " in f" {name} " or code.endswith("T")


def _dedup(cands: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Una entrada por código; nunca ofrecer el mismo artículo dos veces."""
    vistos, out = set(), []
    for c in cands:
        if c["code"] and c["code"] not in vistos:
            vistos.add(c["code"]); out.append(c)
    return out


def _dominancia(cands: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """D5: elimina candidatos ESTRICTAMENTE dominados: B cae si existe A con
    tokens(A) ⊊ tokens(B). Producto vs tester/etiqueta PLV. Solo poda; nunca
    desempata entre conjuntos incomparables (15ml vs 25ml siguen los dos)."""
    toks = [(_canon_tokens(c["name"]), c) for c in cands]
    keep: List[Dict[str, str]] = []
    for tb, b in toks:
        dominado = any(ta < tb for ta, _ in toks)   # subconjunto PROPIO
        if not dominado:
            keep.append(b)
    return keep or cands


def _distintivos(opciones: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], bool]:
    """Nombra cada opción por lo que la DIFERENCIA (A4). Si nada las
    diferencia tras normalizar+léxico -> indistinguibles (D6)."""
    if len(opciones) < 2:
        return opciones, False
    sets = [_canon_tokens(o["name"]) for o in opciones]
    inter = set.intersection(*map(set, sets)) if sets else set()
    indist = all(s == sets[0] for s in sets)
    if indist:
        return opciones, True
    for o, s in zip(opciones, sets):
        diff = [t for t in tokenize(o["name"]) if min(variants(t)) not in inter]
        o["distintivo"] = " ".join(diff[:4])
    return opciones, False


@dataclass
class Resolucion:
    ops: List[Dict[str, Any]] = field(default_factory=list)
    llm_match_usada: bool = False
    log: List[str] = field(default_factory=list)


async def resolver(actos: List[Acto], estado, *, search: Retriever,
                   code_lookup: Optional[CodeLookup],
                   match_call: MatchCall,
                   cantidad_global_pre: Optional[int]) -> Resolucion:
    r = Resolucion()
    ops = r.ops

    # espacios cerrados ------------------------------------------------------
    cart_space = [(l["code"], f"{l.get('name','')} {l['code']}", l) for l in estado.cart]
    prod_items = estado.productos_abiertos()
    conflictos = [i for i in estado.open_items if i.get("tipo") == "conflicto_cantidad"]

    def _opciones_space() -> List[Tuple[str, str, Dict[str, Any]]]:
        sp = []
        for it in prod_items:
            for o in it.get("opciones") or []:
                sp.append((o["code"], f"{o.get('name','')} {o['code']}",
                           {"item": it, "opt": o}))
        return sp

    # separa pedir para retrieval en bloque ----------------------------------
    pedir_actos: List[Acto] = []

    for a in actos:
        if a.tipo == "pedir":
            # Un pedir cuyo texto es SOLO un número que casa con un conflicto de
            # cantidad vivo es en realidad la RESPUESTA a ese conflicto (el 2B
            # suele emitir "13" como pedir, no como elegir). Resolverlo aquí en
            # vez de crear un pedido nuevo de 13 uds (bug 21:54).
            num = a.producto_texto.strip()
            if num.isdigit() and conflictos:
                cf = next((c for c in conflictos
                           if int(num) in (c.get("opciones_qty") or [])), None)
                if cf is not None:
                    ops.append({"op": "elige_qty", "code": cf["code"], "valor": int(num)})
                    continue
            pedir_actos.append(a)

        elif a.tipo == "modificar":
            lig = _ligar(a.objetivo_texto, cart_space)
            if lig.unico is None and not lig.varios and len(estado.cart) == 1:
                lig = Ligado(unico=estado.cart[0])   # "quítalo" con una sola línea
            if lig.unico is None:
                ops.append({"op": "referencia_ambigua", "texto": a.objetivo_texto,
                            "candidatas": [c.get("name") or c["code"] for c in lig.varios]})
                continue
            ln = lig.unico
            ops.append({"op": "modificar", "code": ln["code"], "name": ln.get("name") or "",
                        "modo": a.cantidad.modo, "valor": a.cantidad.valor,
                        "eliminar_con_numeral": a.eliminar_con_numeral,
                        "valor_numeral": a.cantidad.valor})

        elif a.tipo == "elegir":
            # 1) ¿resuelve un conflicto de cantidad? (respuesta "19")
            num = a.objetivo_texto.strip()
            if num.isdigit() and conflictos:
                cf = next((c for c in conflictos if int(num) in (c.get("opciones_qty") or [])),
                          None)
                if cf is not None:
                    ops.append({"op": "elige_qty", "code": cf["code"], "valor": int(num)})
                    continue
            # 2) ¿un código tecleado de entre las opciones?
            typed = codes_in_text(a.objetivo_texto)
            opt_space = _opciones_space()
            if typed:
                hitk = next(((k, obj) for k, _, obj in opt_space if k in typed), None)
                if hitk:
                    it, o = hitk[1]["item"], hitk[1]["opt"]
                    ops.append({"op": "elige", "item_texto": it["texto"],
                                "code": o["code"], "name": o.get("name") or "",
                                "brand": o.get("brand") or "", "fuente": o.get("fuente") or ""})
                    continue
                # código fuera de las opciones -> se trata como pedir por código
                pedir_actos.append(Acto(tipo="pedir", evidencia=a.evidencia,
                                        producto_texto=a.objetivo_texto,
                                        cantidad=a.cantidad))
                continue
            # 3) por descripción contra las opciones vivas
            lig = _ligar(a.objetivo_texto, opt_space)
            if lig.unico is not None:
                it, o = lig.unico["item"], lig.unico["opt"]
                ops.append({"op": "elige", "item_texto": it["texto"],
                            "code": o["code"], "name": o.get("name") or "",
                            "brand": o.get("brand") or "", "fuente": o.get("fuente") or ""})
            else:
                ops.append({"op": "referencia_ambigua", "texto": a.objetivo_texto,
                            "candidatas": [v["opt"].get("name") or v["opt"]["code"]
                                           for v in lig.varios]})

        elif a.tipo == "rechazar":
            objetivo = None
            if a.objetivo_texto:
                lig = _ligar(a.objetivo_texto,
                             [(i["texto"], i["texto"], i) for i in prod_items])
                if lig.unico is not None:
                    objetivo = lig.unico["texto"]
            ops.append({"op": "rechazo", "item_texto": objetivo})

        elif a.tipo == "cantidad_global":
            ops.append({"op": "cantidad_global", "valor": a.valor})

        elif a.tipo == "repetir_ultimo":
            ops.append({"op": "repetir_ultimo",
                        "qty_override": a.cantidad.valor if a.cantidad.modo == "total" else None})

        elif a.tipo in ("confirmar", "cerrar"):
            ops.append({"op": a.tipo})

        elif a.tipo == "otro":
            ops.append({"op": "otro", "subtipo": a.subtipo})

    # ---- pedir que en realidad elige una opción viva por código ------------
    #      ("El LML58752" puede llegar como pedir; si el código está entre las
    #      opciones vivas, es un elegir — hereda cantidad y cierra el ítem.)
    opt_space_now = _opciones_space()
    pedir_filtrados: List[Acto] = []
    for a in pedir_actos:
        typed = codes_in_text(a.producto_texto)
        hitk = next(((k, obj) for k, _, obj in opt_space_now if k in typed), None) \
            if typed else None
        if hitk is not None:
            it, o = hitk[1]["item"], hitk[1]["opt"]
            if a.cantidad.valor is not None:
                it["qty"] = a.cantidad.valor
            ops.append({"op": "elige", "item_texto": it["texto"],
                        "code": o["code"], "name": o.get("name") or "",
                        "brand": o.get("brand") or "", "fuente": o.get("fuente") or ""})
        else:
            pedir_filtrados.append(a)
    pedir_actos = pedir_filtrados

    # ---- pedir: enriquecimiento de open_items (caso 3.9) -------------------
    restantes: List[Acto] = []
    for a in pedir_actos:
        lig = _ligar(a.producto_texto, [(i["texto"], i["texto"], i) for i in prod_items])
        it = lig.unico
        if it is not None:
            # hereda: cantidad ya dicha y rechazos persistentes
            if a.cantidad.valor is not None:
                it["qty"] = a.cantidad.valor
            a._open_item = it        # type: ignore[attr-defined]
        restantes.append(a)

    # ---- pedir: códigos tecleados (D4) --------------------------------------
    con_texto: List[Acto] = []
    typed_codes: Dict[str, Acto] = {}
    for a in restantes:
        codes = codes_in_text(a.producto_texto) or codes_in_text(a.evidencia)
        only_code = codes and len(tokenize(a.producto_texto)) <= 1
        if codes and only_code:
            typed_codes[codes[0]] = a
        else:
            con_texto.append(a)

    if typed_codes and code_lookup is not None:
        found = await code_lookup(list(typed_codes.keys()))
        for code, a in typed_codes.items():
            row = (found or {}).get(code)
            if row:
                ops.append({"op": "add", "code": row.get("code") or code,
                            "name": row.get("name") or "", "brand": row.get("brand") or "",
                            "qty": a.cantidad.valor, "fuente": "codigo_cliente"})
            else:
                ops.append({"op": "codigo_no_existe", "code": code})
        typed_codes = {}
    # sin code_lookup cableado: los códigos tecleados van por retrieval igual
    con_texto.extend(typed_codes.values())

    # ---- pedir: retrieval en bloque ----------------------------------------
    ambiguos: List[Dict[str, Any]] = []
    if con_texto:
        queries = [a.producto_texto for a in con_texto]
        results = await search(queries)
        for a in con_texto:
            hits = results.get(a.producto_texto) or []
            it_prev = getattr(a, "_open_item", None)
            qty = a.cantidad.valor if a.cantidad.valor is not None else \
                (it_prev or {}).get("qty")
            rechazados = set((it_prev or {}).get("rechazados") or [])

            full = [h for h in hits if h.get("covers_query")]
            cands = _dedup(_dominancia([_hit_opt(h) for h in (full or hits)]))
            cands = [c for c in cands if c["code"] and c["code"] not in rechazados]
            # gemelos de nombre (mismo producto, varios EAN) -> el más nuevo,
            # sin preguntar. Salvo que el cliente teclee un código concreto:
            # entonces ese código (si está en los hits/activo) manda -> se
            # conserva el grupo para que el match exacto de abajo lo elija.
            typed_here = set(codes_in_text(a.producto_texto))
            if not (typed_here & {c["code"] for c in cands}):
                cands = _colapsar_gemelos(cands)
            # Testers fuera salvo que TODO lo hallado sea tester (o el cliente
            # los pidiera explícitamente en el texto).
            pidio_tester = "tester" in (a.producto_texto or "").lower() or \
                           "test" in (a.producto_texto or "").lower()
            if not pidio_tester:
                sin_test = [c for c in cands if not _es_tester(c)]
                if sin_test:
                    cands = sin_test
            # Si el cliente dijo un número de referencia (403, 402...), ese
            # número es un discriminador estructural: descarta candidatos cuyo
            # nombre NO lo contiene (el tester 401 y el insta-micro 402 sobran
            # ante "brow pencil 403"). Solo poda si con ello queda >=1.
            nums_ref = {t for t in re.split(r"[^0-9]+", a.producto_texto) if len(t) == 3}
            if nums_ref:
                filtrados = [c for c in cands
                             if any(n in c["name"] for n in nums_ref)]
                if filtrados:
                    cands = filtrados

            # código exacto tecleado dentro del texto y presente en hits
            typed = set(codes_in_text(a.producto_texto))
            exact = next((c for c in cands if c["code"] in typed), None)

            if not cands:
                ops.append({"op": "sin_candidatos", "texto": a.producto_texto, "qty": qty})
            elif exact is not None:
                if it_prev is not None:
                    ops.append({"op": "elige", "item_texto": it_prev["texto"],
                                "code": exact["code"], "name": exact["name"],
                                "brand": exact["brand"], "fuente": exact["fuente"]})
                else:
                    ops.append({"op": "add",
                                **{k: exact[k] for k in ("code", "name", "brand", "fuente")},
                                "qty": qty})
            elif len(cands) == 1 and full:
                c = cands[0]
                if it_prev is not None:
                    ops.append({"op": "elige", "item_texto": it_prev["texto"],
                                "code": c["code"], "name": c["name"],
                                "brand": c["brand"], "fuente": c["fuente"]})
                else:
                    ops.append({"op": "add", **{k: c[k] for k in ("code", "name", "brand", "fuente")},
                                "qty": qty})
            else:
                opciones, indist = _distintivos(cands[:MAX_OPCIONES])
                if indist or not full:
                    # D6 / cobertura parcial: pregunta directa, sin match.
                    ops.append({"op": "opciones", "texto": a.producto_texto, "qty": qty,
                                "opciones": opciones, "indistinguibles": indist})
                else:
                    ambiguos.append({"texto": a.producto_texto, "qty": qty,
                                     "opciones": opciones, "item_prev": it_prev})

    # ---- match por lotes (una llamada, tope) --------------------------------
    if ambiguos:
        lote = ambiguos[:MATCH_MAX_ITEMS]
        resto = ambiguos[MATCH_MAX_ITEMS:]
        payload = [{"i": i, "texto": x["texto"],
                    "opciones": [o["name"] for o in x["opciones"]]}
                   for i, x in enumerate(lote)]
        verdicts = await match_call(payload)
        r.llm_match_usada = verdicts is not None
        vmap = {v.get("item"): v for v in (verdicts or []) if isinstance(v, dict)}
        for i, x in enumerate(lote):
            v = vmap.get(i) or {"veredicto": "varios"}
            idx = v.get("indice")
            if v.get("veredicto") == "unico" and isinstance(idx, int) \
                    and 0 <= idx < len(x["opciones"]):
                c = x["opciones"][idx]
                if x.get("item_prev") is not None:
                    ops.append({"op": "elige", "item_texto": x["item_prev"]["texto"],
                                "code": c["code"], "name": c["name"],
                                "brand": c["brand"], "fuente": c.get("fuente") or ""})
                else:
                    ops.append({"op": "add", "code": c["code"], "name": c["name"],
                                "brand": c["brand"], "fuente": c.get("fuente") or "",
                                "qty": x["qty"]})
            elif v.get("veredicto") == "ninguno":
                ops.append({"op": "sin_candidatos", "texto": x["texto"], "qty": x["qty"]})
            else:
                ops.append({"op": "opciones", "texto": x["texto"], "qty": x["qty"],
                            "opciones": x["opciones"], "indistinguibles": False})
        for x in resto:   # degradación determinista: solo puede producir preguntas
            ops.append({"op": "opciones", "texto": x["texto"], "qty": x["qty"],
                        "opciones": x["opciones"], "indistinguibles": False})

    return r

# ============================================================================
# RENDER — Etapa 4, la plantilla es la respuesta
# ============================================================================


BOT_SIGNATURE = os.getenv("AGENT_BOT_SIGNATURE", "— Kapa 🤖")
COLAPSO_UMBRAL = int(os.getenv("PIPE_COLAPSO_UMBRAL", "2"))   # >N sin resolver -> colapsa

_RE_MD = re.compile(r"[*_#`]|^\s*[-•]\s", re.M)
_RE_DIGITS = re.compile(r"\d+")


def _n(name: str, code: str) -> str:
    return (name or "").strip().title() or code


def _linea_producto(name: str, qty: int, code: str, marcado: bool = False) -> str:
    star = " *" if marcado else ""
    return f"{_n(name, code)}, {qty} uds ({code}){star}"


def _saludo(intro_mode: str, client_name: str, salesman_name: str) -> str:
    if intro_mode == "new":
        return (f"¡Hola{', ' + client_name.split()[0].title() if client_name and client_name != 'Cliente' else ''}! "
                f"Soy Kapa, el asistente de pedidos de {salesman_name}. "
                "Te ayudo a dejar tu pedido apuntado y él lo confirma.")
    if intro_mode == "renew":
        return f"¡Hola de nuevo! Soy Kapa, el asistente de {salesman_name}."
    return ""


def render(hechos: List[Hecho], estado: Estado, *, intro_mode: str,
           client_name: str, salesman_name: str, commercial_turn: bool) -> str:
    """hechos -> texto plano WhatsApp-safe. Determinista."""
    conf: List[str] = []
    preguntas: List[str] = []
    otras: List[str] = []
    cortesias: List[str] = []
    cierre: Optional[str] = None
    derivacion: Optional[str] = None

    # colapso (inv. 9): cuenta ítems que acabarán pidiendo elección/datos
    pend = [h for h in hechos if h.tipo in ("pregunta_opciones", "sin_candidatos",
                                            "rechazo_registrado", "recordatorio_pendiente",
                                            "referencia_ambigua")]
    colapsar = len(pend) > COLAPSO_UMBRAL

    for h in hechos:
        d = h.datos
        if h.tipo == "confirmado":
            conf.append("Apuntado: " + _linea_producto(d["name"], d["qty"], d["code"],
                                                       d.get("marcado", False)))
        elif h.tipo == "actualizado":
            conf.append("Actualizado: " + _linea_producto(d["name"], d["qty"], d["code"]))
        elif h.tipo == "sin_cambio_confirmado":
            conf.append("Eso ya lo tienes en el pedido: "
                        + _linea_producto(d["name"], d["qty"], d["code"]) + ".")
        elif h.tipo == "eliminado":
            conf.append(f"Quitado del pedido: {_n(d['name'], d['code'])} ({d['code']}).")

        elif h.tipo == "pregunta_opciones" and not colapsar:
            ops = d.get("opciones") or []
            if d.get("indistinguibles"):
                cods = ", ".join(o["code"] for o in ops)
                preguntas.append(
                    f"De \"{d['texto']}\" tengo varias referencias con exactamente el mismo "
                    f"nombre, así que solo puedo distinguirlas por código: {cods}. "
                    "¿Cuál de esos códigos es el tuyo?")
            else:
                partes = []
                for o in ops:
                    tag = o.get("distintivo") or _n(o.get("name", ""), o["code"])
                    partes.append(f"{tag.strip().lower()} ({o['code']})")
                qty_txt = f" (me dijiste {d['qty']} uds)" if d.get("qty") else ""
                preguntas.append(
                    f"De \"{d['texto']}\"{qty_txt} tengo " + " y ".join(partes) +
                    ". ¿Cuál quieres?")
        elif h.tipo == "sin_candidatos" and not colapsar:
            preguntas.append(
                f"\"{d['texto']}\" no lo encuentro entre lo que sueles pedir. "
                "¿Me pasas el nombre completo o el código? Si no, se lo dejo a "
                f"{salesman_name} para que lo mire.")
        elif h.tipo == "rechazo_registrado" and not colapsar:
            preguntas.append(
                f"Entendido, esas opciones de \"{d['texto']}\" no son. "
                "Pásame el código o el nombre completo del que buscas y lo apunto.")
        elif h.tipo == "recordatorio_pendiente" and not colapsar:
            preguntas.append(f"Sigo con \"{d['item']}\" pendiente: dime el código o el "
                             "nombre completo cuando puedas.")

        elif h.tipo == "conflicto_cantidad":
            nums = " , ".join(str(x) for x in d.get("opciones_qty") or [])
            preguntas.append(
                f"De {_n(d['name'], d['code'])} ({d['code']}) me has dado dos cantidades "
                f"distintas. ¿Con cuál me quedo: {nums}?")
        elif h.tipo == "pregunta_baja":
            preguntas.append(f"Con eso {_n(d['name'], d['code'])} ({d['code']}) se quedaría "
                             "en 0. ¿Te lo quito del pedido?")
        elif h.tipo == "pregunta_eliminar_o_restar":
            preguntas.append(f"De {_n(d['name'], d['code'])} ({d['code']}): ¿quito la línea "
                             f"entera o resto {d.get('n') or 1}?")
        elif h.tipo == "referencia_ambigua" and not colapsar:
            cands = list(dict.fromkeys(d.get("candidatas") or []))   # dedup, orden
            if cands:
                preguntas.append(
                    f"De \"{d['texto']}\" no sé cuál quieres: "
                    + " o ".join(cands[:3]) + ". ¿Me lo confirmas?")
            else:
                preguntas.append(
                    f"Perdona, no acabo de encontrar \"{d['texto']}\". "
                    "¿Me das la marca o el código y lo apunto?")
        elif h.tipo == "codigo_no_existe":
            otras.append(f"El código {d['code']} no me consta en el sistema; puede ser "
                         f"antiguo o de otro proveedor. Se lo paso a {salesman_name} "
                         "para que lo revise.")

        elif h.tipo == "propuesta":
            items = d.get("items") or []
            lineas = "; ".join(_linea_producto(i.get("name", ""), int(i.get("qty") or 1),
                                               i.get("code", "")) for i in items)
            if d.get("tipo") == "propuesta_repetir":
                preguntas.append(f"Como tu último pedido quedaría: {lineas}. "
                                 "¿Lo apunto así?")
            elif d.get("tipo") == "propuesta_media":
                preguntas.append(f"De lo que me has mandado he sacado: {lineas}. "
                                 "Al venir de una foto/audio prefiero confirmarlo: ¿lo apunto?")
            elif d.get("tipo") == "propuesta_cierre":
                preguntas.append(f"Tu pedido queda: {lineas}. ¿Lo confirmo y se lo paso a "
                                 f"{salesman_name}?")
        elif h.tipo == "cerrado":
            items = d.get("cart") or []
            lineas = "; ".join(_linea_producto(i.get("name", ""), int(i.get("qty") or 1),
                                               i.get("code", "")) for i in items)
            cierre = (f"¡Pedido confirmado! {lineas}. Se lo paso ahora a {salesman_name}, "
                      "que lo valida antes del envío. Gracias.")

        elif h.tipo == "derivacion":
            if d.get("motivo") == "producto_no_encontrado":
                cual = f" \"{d['texto']}\"" if d.get("texto") else ""
                derivacion = (f"No consigo localizar{cual} entre lo que tenemos. Se lo paso a "
                              f"{salesman_name}, que se conectará en cuanto pueda para ayudarte con eso.")
            else:
                derivacion = (f"De eso te informa {salesman_name} directamente; le aviso para "
                              "que te escriba en cuanto pueda.")
        elif h.tipo == "aviso_comercial":
            derivacion = (f"Aviso a {salesman_name} para que se conecte lo antes posible "
                          "y te atienda.")
        elif h.tipo == "opt_out":
            derivacion = (f"Entendido, no te escribo más. Aviso a {salesman_name} para que "
                          "te atienda personalmente.")
        elif h.tipo == "recap_pedido":
            if estado.cart:
                lineas = "; ".join(_linea_producto(l.get("name", ""), int(l["qty"]), l["code"])
                                   for l in estado.cart)
                otras.append(f"Ahora mismo llevas apuntado: {lineas}. ¿Añades algo más o lo cerramos?")
            else:
                otras.append("Todavía no tienes nada apuntado en este pedido. ¿Qué te pongo?")
        elif h.tipo == "cortesia":
            st = d.get("subtipo")
            if st == "confirmacion_sin_pregunta":
                cortesias.append("¡De acuerdo!")
            elif st in ("saludo",):
                # si el turno ya lleva saludo de presentación, no dupliques "¡Hola!"
                habra_intro = (intro_mode == "new" or (intro_mode == "renew" and commercial_turn))
                if not habra_intro:
                    cortesias.append("¡Hola!")
            elif st in ("agradecimiento", "despedida"):
                cortesias.append("¡Gracias a ti! Aquí estoy si necesitas apuntar algo más.")
            elif st == "pregunta_bot":
                cortesias.append("Soy Kapa, un asistente: apunto tu pedido y "
                                 f"{salesman_name} lo confirma antes de enviar nada.")
        elif h.tipo == "sin_entender":
            preguntas.append("No estoy seguro de haberte entendido. Dime producto y "
                             "cantidad (mejor con el código si lo tienes) y lo apunto.")

    if colapsar:
        textos = sorted({(h.datos.get("texto") or h.datos.get("item") or "").strip()
                         for h in pend if (h.datos.get("texto") or h.datos.get("item"))})
        preguntas.append(
            "Me quedan varios artículos por concretar (" + ", ".join(f"\"{t}\"" for t in textos)
            + "). Para no marearte con opciones: ¿me pasas el nombre completo o el código "
              "de cada uno?")

    bloques: List[str] = []
    sal = _saludo(intro_mode, client_name, salesman_name) if \
        (intro_mode == "new" or (intro_mode == "renew" and commercial_turn)) else ""
    if sal:
        bloques.append(sal)
    if cierre:
        bloques.extend(conf)
        bloques.append(cierre)
    else:
        bloques.extend(conf)
        bloques.extend(preguntas)
        bloques.extend(otras)
    if derivacion:
        bloques.append(derivacion)
    if not any((conf, preguntas, otras, cierre, derivacion)) and cortesias:
        bloques.extend(cortesias)
    elif cortesias and not (conf or preguntas or cierre):
        bloques.extend(cortesias)

    cuerpo = "\n".join(b for b in bloques if b).strip()
    if not cuerpo:
        return ""
    return cuerpo + "\n" + BOT_SIGNATURE


# ============================================================================
# TURNO — orquestador, llamadas LLM y contrato con handlers
# ============================================================================


MAX_LLM_CALLS = int(os.getenv("PIPE_MAX_LLM_CALLS", "4"))
EXTRACT_TIMEOUT_S = float(os.getenv("PIPE_EXTRACT_TIMEOUT_S", "90"))
MATCH_TIMEOUT_S = float(os.getenv("PIPE_MATCH_TIMEOUT_S", "60"))
EXTRACT_MAX_TOKENS = int(os.getenv("PIPE_EXTRACT_MAX_TOKENS", "700"))
MATCH_MAX_TOKENS = int(os.getenv("PIPE_MATCH_MAX_TOKENS", "300"))
MSG_MAX_CHARS = int(os.getenv("PIPE_MSG_MAX_CHARS", "2400"))

_client: Optional[CimaClient] = None


# --------------------------------------------------------------------- result
@dataclass
class AgentResult:
    """Contrato con handle_ai_trigger, sin cambios. reply == "" es silencio."""
    reply: str = ""
    order_status: str = "IDLE"
    cart: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    handoff: bool = False
    opt_out: bool = False
    silent: bool = False
    open_items: List[Dict[str, Any]] = field(default_factory=list)
    guide_shown: bool = False
    ctx: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- LLM helpers
class _Budget:
    def __init__(self, limit: int = MAX_LLM_CALLS) -> None:
        self.limit = limit
        self.used = 0
        self.detail: List[Dict[str, Any]] = []

    def can(self) -> bool:
        return self.used < self.limit

    def spend(self, kind: str, ms: int, ok: bool) -> None:
        self.used += 1
        self.detail.append({"kind": kind, "ms": ms, "ok": ok})


async def _chat_json(system: str, user: str, schema: Dict[str, Any], *,
                     max_tokens: int, timeout: float, temperature: float,
                     budget: _Budget, kind: str,
                     fmt: Any = None) -> Optional[Dict[str, Any]]:
    """Una llamada que devuelve JSON. `fmt` por defecto es el schema (cima
    puede restringir con él); pasar fmt="json" fuerza el modo laxo, para
    cuando el motor NO respeta el schema-dict y hay que apoyarse en el prompt.
    None = fallo (HTTP o JSON ilegible)."""
    if not budget.can():
        logger.warning(f"[PIPE] presupuesto agotado; se omite {kind}")
        return None
    t0 = time.monotonic()
    ok = False
    try:
        text = await asyncio.to_thread(inference.chat, f"{system}\n\n{user}", timeout=timeout)
        data = inference.json_from(text)
        ok = data is not None
        return data
    except Exception as e:
        logger.warning(f"[PIPE] {kind} falló: {e}")
        return None
    finally:
        ms = int((time.monotonic() - t0) * 1000)
        budget.spend(kind, ms, ok)
        logger.info(f"[PIPE] {kind} {'ok' if ok else 'FAIL'} in {ms}ms "
                    f"({budget.used}/{budget.limit})")


# ------------------------------------------------------------ Etapa 1: prompt
# D9: ningún literal con forma de código de producto en el texto de
# instrucción ni en los ejemplos. Los códigos que aparezcan en el bloque de
# CONTEXTO (carrito/opciones) son DATOS recuperados, no ejemplos.
_EXTRACT_SYSTEM = """Eres el intérprete de mensajes de Kapa, un asistente que SOLO monta pedidos de cosmética profesional por WhatsApp mientras el comercial no está. Tu única tarea: convertir el mensaje del cliente en una lista de actos JSON. No respondes al cliente, no eliges productos, no calculas.

FRONTERA — decide primero de qué lado cae cada intención:
· LO QUE KAPA ATIENDE (todo lo de montar el pedido): añadir productos, cambiar cantidades, quitar líneas, elegir/rechazar opciones, LEER el pedido actual, repetir un pedido anterior, cerrar el pedido. Esto se mapea a los tipos pedir/modificar/elegir/rechazar/confirmar/cerrar/repetir_ultimo/cantidad_global y a otro(estado_pedido). El estado del pedido lo sabe Kapa: NUNCA se deriva.
· LO QUE KAPA DERIVA AL COMERCIAL (no lo puede atender): precios, descuentos, ofertas, stock/disponibilidad, plazos de entrega, envíos, facturas, pagos → otro(precio_stock_factura). Detalles o especificaciones del producto (ingredientes, uso, tono exacto, para qué sirve) → otro(especificacion). Quejas, reclamaciones, devoluciones → otro(queja). Cualquier otra cosa que no sea montar un pedido → otro(fuera_ambito).
· LO QUE HACE QUE KAPA CALLE Y AVISE AL COMERCIAL: el cliente exige hablar con el comercial o está enfadado → otro(escalada). No quiere hablar con un asistente → otro(rechazo_bot).
Ante la duda entre atender y derivar: si es sobre QUÉ productos van en el pedido, es de Kapa; si es sobre condiciones, detalles, dinero o problemas, es del comercial.

TIPOS de acto:
- pedir: el cliente quiere un artículo. producto_texto = sus palabras. Si dijo un número de unidades EN ESA LÍNEA: cantidad_modo="total", cantidad_valor=ese número. Si no dijo número: cantidad_modo="sin_cambio".
- modificar: cambia una línea que ya está en EL PEDIDO ACTUAL. objetivo_texto = cómo la nombra. cantidad_modo: "total" (pon N / N en vez de M / deja N / que queden N), "incremento" (añade/suma N más), "decremento" (quita/resta N), "eliminar" (quita la línea / ya no lo quiero / bórralo).
- elegir: responde a una pregunta pendiente indicando cuál quiere (por descripción, por código o con un número si la pregunta era de cantidades). objetivo_texto = sus palabras.
- rechazar: dice que las opciones ofrecidas NO son lo que busca pero SIGUE queriendo el artículo ("ese no es", "ninguno de esos"). objetivo_texto = el artículo si lo nombra, si no vacío.
- confirmar: respuesta afirmativa a la última pregunta del asistente ("sí", "vale", "ok", "perfecto", "eso es"). OJO: "apunta"/"ponme"/"anota" seguido de productos NO es confirmar, es parte de pedir.
- cerrar: da el pedido por terminado y quiere enviarlo ("eso es todo", "mándalo", "confírmame el pedido", "ciérralo").
- repetir_ultimo: pide lo mismo que el último pedido. Si dice "N de cada": cantidad_modo="total", cantidad_valor=N.
- cantidad_global: una frase suelta que fija N unidades para todo lo anterior ("6 de cada uno", "todo a 6"). valor=N.
- otro: nada de lo anterior. subtipo según la FRONTERA de arriba: estado_pedido | precio_stock_factura | especificacion | queja | fuera_ambito | escalada | rechazo_bot | saludo | despedida | agradecimiento | pregunta_bot (pregunta qué eres).

REGLAS:
1. Un acto por intención; un mensaje puede tener varios actos.
2. evidencia = el fragmento LITERAL del mensaje del que sale el acto. Cópialo tal cual.
3. cantidad_valor SOLO puede ser un número que el cliente escribió. Nunca lo inventes ni lo calcules.
4. Cada línea de una lista es su propio acto pedir, con SU cantidad. No mezcles cantidades entre líneas.
5. "quita uno de X" es modificar decremento 1, NO eliminar. "quita X" (sin número) es eliminar.
6. Si el mensaje nombra algo que aparece en PENDIENTES, suele ser elegir o rechazar, no pedir.
7. LEE LA CONVERSACIÓN. Los mensajes cortos casi siempre se refieren a lo último que dijo el Asistente:
   - "sí", "ese", "el segundo", "vale" tras una pregunta de opciones → elegir (objetivo_texto = lo que señala).
   - un número suelto ("13") tras una pregunta de cantidades → elegir (objetivo_texto = ese número).
   - "no, ese no", "ninguno" → rechazar.
   - "quítame el otro", "agrégame 2 más" → modificar sobre la línea del PEDIDO ACTUAL a la que se refieren.
   Sin la conversación no se pueden entender; úsala.
8. No respondas nada fuera del JSON.
9. "hablar": es false cuando el bot NO debe meterse. Dos casos: (a) charla no comercial suelta ("ok", "gracias", un comentario personal) que no responde a ninguna pregunta pendiente; (b) el cliente y el COMERCIAL ya están en una conversación que NO es de pedido —negociando precios, detalles del producto, una incidencia— y el mensaje continúa ESA charla (lo ves en el hilo: hay mensajes recientes del Comercial sobre el tema). En ese caso el bot no se interpone. Es true SIEMPRE que haya algún acto de pedido, una respuesta a una pregunta del asistente, o el cliente pida hablar con el comercial. Ante la duda, true.
10. Si el cliente pide EXPLÍCITAMENTE hablar con el comercial ("que me llame", "quiero hablar con {comercial}", "pásame con una persona") -> otro(pide_comercial). Eso NO es silencio: el bot avisa al comercial.

EJEMPLOS (los nombres son inventados):
Mensaje: "ponme 12 del serum facial noche"
actos: [pedir(producto_texto="serum facial noche", cantidad_modo="total", cantidad_valor=12, evidencia="ponme 12 del serum facial noche")]
Mensaje: "quiero 3 de la mascara volumen"
actos: [pedir(producto_texto="mascara volumen", cantidad_modo="total", cantidad_valor=3, evidencia="quiero 3 de la mascara volumen")]
Mensaje: "ponme 6 de la crema de manos aloe y quita una del esmalte rojo, mejor dime tambien precios"
actos: [pedir(producto_texto="crema de manos aloe", cantidad_modo="total", cantidad_valor=6, evidencia="ponme 6 de la crema de manos aloe"), modificar(objetivo_texto="el esmalte rojo", cantidad_modo="decremento", cantidad_valor=1, evidencia="quita una del esmalte rojo"), otro(subtipo="precio_stock_factura", evidencia="dime tambien precios")]
Casi todo mensaje del cliente contiene al menos un acto. Devolver una lista vacía es casi siempre un error: si menciona un producto, es pedir."""


def _contexto_extraccion(estado: Estado, mensaje: str, *,
                         resumen: str = "", hilo: str = "") -> str:
    partes: List[str] = []
    if resumen.strip():
        partes.append("RESUMEN DE LA CONVERSACIÓN:")
        partes.append(resumen.strip())
    if hilo.strip():
        partes.append("CONVERSACIÓN RECIENTE (Cliente / Comercial / Asistente):")
        partes.append(hilo.strip())
    if estado.cart:
        partes.append("PEDIDO ACTUAL:")
        for i, l in enumerate(estado.cart, 1):
            partes.append(f"{i}. {l.get('name') or l['code']} x{l['qty']} ({l['code']})")
    else:
        partes.append("PEDIDO ACTUAL: vacío")
    pend: List[str] = []
    for it in estado.open_items:
        t = it.get("tipo")
        if t == "producto":
            ops = ", ".join(f"{o.get('name')} ({o.get('code')})"
                            for o in it.get("opciones") or []) or "sin opciones aún"
            q = f", cantidad dicha: {it['qty']}" if it.get("qty") else ""
            pend.append(f"- \"{it.get('texto')}\"{q} → opciones: {ops}")
        elif t == "conflicto_cantidad":
            pend.append(f"- {it.get('name')} ({it.get('code')}): cantidades posibles "
                        f"{it.get('opciones_qty')}")
        else:
            pend.append(f"- pregunta sí/no pendiente: {t.replace('propuesta_', '')}")
    partes.append("PENDIENTES:" if pend else "PENDIENTES: ninguno")
    partes.extend(pend)
    partes.append("MENSAJE DEL CLIENTE:")
    partes.append(mensaje[:MSG_MAX_CHARS])
    return "\n".join(partes)


# Formato de salida escrito para el modo laxo (cuando el schema no restringe).
_EXTRACT_FORMATO = (
    '\n\nResponde SOLO con este JSON, sin texto alrededor. "actos" es SIEMPRE '
    'una lista (aunque haya un solo acto). "hablar" es true si el bot debe '
    'responder este turno, false si NO hace falta responder (ver regla 9):\n'
    '{"hablar": true, "actos": [{"tipo": "pedir", "producto_texto": "...", '
    '"cantidad_modo": "total", "cantidad_valor": 0, "evidencia": "..."}]}')


async def _extraer(estado: Estado, mensaje: str, budget: _Budget, *,
                   resumen: str = "", hilo: str = "") -> Tuple[Optional[List[Acto]], bool]:
    # COMPRENDER: única llamada que ve la CONVERSACIÓN (resumen + hilo con
    # roles). Devuelve (actos, hablar). Modo laxo porque cima ignora el schema.
    user = _contexto_extraccion(estado, mensaje, resumen=resumen, hilo=hilo) + _EXTRACT_FORMATO
    data = await _chat_json(_EXTRACT_SYSTEM, user, ACTOS_SCHEMA,
                            max_tokens=EXTRACT_MAX_TOKENS, timeout=EXTRACT_TIMEOUT_S,
                            temperature=0.0, budget=budget, kind="extract", fmt="json")
    if data is None:
        return None, True                      # fallo -> por defecto, hablar
    hablar = bool(data.get("hablar", True)) if isinstance(data, dict) else True
    actos = validar_actos(parse_actos(data), mensaje)
    if actos:
        logger.info("[PIPE] actos: " + "; ".join(a.as_log() for a in actos))
    else:
        logger.info(f"[PIPE] actos vacíos; crudo={json.dumps(data)[:300]}")
    return actos, hablar


# ------------------------------------------------------------ Etapa 2: match
_MATCH_SYSTEM = """Eres el seleccionador de productos de un asistente de pedidos. Para cada item te doy lo que escribió el cliente y una lista numerada de candidatos del catálogo. Decide por item:
- "unico" + indice: exactamente UN candidato es claramente lo que pide.
- "varios": más de uno podría serlo (ante la duda, elige esto).
- "ninguno": ninguno encaja con lo que pide.
Los índices empiezan en 0. No inventes índices fuera de la lista.
Responde SOLO con este JSON (una entrada por item):
{"items": [{"item": 0, "veredicto": "unico", "indice": 2}]}"""


def _match_user(payload: List[Dict[str, Any]]) -> str:
    partes = []
    for it in payload:
        partes.append(f"ITEM {it['i']}: cliente escribió \"{it['texto']}\"")
        for j, name in enumerate(it["opciones"]):
            partes.append(f"  {j}. {name}")
    return "\n".join(partes)


def _actos_estructurales(texto: str, estado: Estado) -> List[Acto]:
    """Cuando la extracción no ve ningún acto, lo puramente ESTRUCTURAL no
    puede perderse por ello (el LLM interpreta lenguaje; los códigos son
    forma): un token con forma de código se convierte en `pedir` (la
    resolución ya lo liga a la opción viva, con herencia de cantidad, o lo
    busca), y un número suelto que casa con un conflicto de cantidad vivo se
    convierte en `elegir`. Nada más: el resto sí es semántica y sigue siendo
    territorio del modelo."""
    out: List[Acto] = []
    for c in codes_in_text(texto):
        out.append(Acto(tipo="pedir", producto_texto=c, evidencia=c))
    limpio = re.sub(r"[^0-9]", "", texto) if len(texto.strip()) <= 12 else ""
    if limpio.isdigit() and limpio:
        n = int(limpio)
        if any(n in (i.get("opciones_qty") or []) for i in estado.open_items
               if i.get("tipo") == "conflicto_cantidad"):
            out.append(Acto(tipo="elegir", objetivo_texto=limpio, evidencia=limpio))
    return out


# ------------------------------------------------------------------ run_agent
async def run_agent(*, client_name: str, salesman_name: str,
                    session: Dict[str, Any], recent_history: str,
                    current_message: str, intro_mode: str,
                    search: Retriever,
                    code_lookup: Optional[CodeLookup] = None) -> AgentResult:
    """Un turno completo (handlers.run_turn).
    `recent_history` llega con roles (Cliente/Comercial/Asistente) y ENTRA en
    el prompt de COMPRENDER: es lo que permite resolver "sí ese", "13",
    "quítame el otro" leyendo lo que se dijo antes."""
    t_turn = time.monotonic()
    budget = _Budget()
    estado_in = Estado.from_session(session)
    guide_shown = bool(session.get("guide_shown"))
    resumen = ""          # el hilo reciente basta para un chat de pedido
    hilo = recent_history or ""

    def _result(reply: str, estado: Estado, hechos: List[Hecho],
                extra_ctx: Optional[Dict[str, Any]] = None) -> AgentResult:
        handoff = any(h.tipo in ("derivacion", "escalada", "aviso_comercial") for h in hechos)
        opt_out = any(h.tipo == "opt_out" for h in hechos)
        ctx = {"pipeline": "v2", "llm_calls": budget.detail,
               "turn_ms": int((time.monotonic() - t_turn) * 1000),
               **(extra_ctx or {})}
        return AgentResult(reply=reply, order_status=estado.order_status,
                           cart=estado.cart, summary="",
                           handoff=handoff, opt_out=opt_out, silent=False,
                           open_items=estado.open_items,
                           guide_shown=guide_shown, ctx=ctx)

    # -------- Etapa 0
    pre = preprocesar(current_message)
    if pre.log:
        logger.info("[PIPE] pre: " + ", ".join(pre.log))
    if not pre.texto and not pre.pedidos and pre.cantidad_global is None \
            and not pre.fuente_media:
        return _result("", estado_in, [])   # nada que responder

    # renglones de lista consumidos en la Etapa 0: actos construidos por
    # código, con su cantidad de SU línea (los casos 1-2 por construcción)
    actos_lista = [Acto(tipo="pedir", producto_texto=n, evidencia=orig,
                        cantidad=(Cantidad("total", q) if q is not None else Cantidad()))
                   for n, q, orig in pre.pedidos]

    # -------- Etapa 1 COMPRENDER (solo si queda LENGUAJE que interpretar)
    hablar = True
    if pre.texto:
        actos, hablar = await _extraer(estado_in, pre.texto, budget, resumen=resumen, hilo=hilo)
    else:
        actos = []
    if not actos and not actos_lista:   # ilegible dos veces O schema-válido pero vacío
        actos = _actos_estructurales(pre.texto or current_message, estado_in) or actos
        if actos:
            logger.info("[PIPE] salvamento estructural: "
                        + "; ".join(a.as_log() for a in actos))
    if actos is None and actos_lista:
        actos = []          # la lista se procesa aunque la extracción muriera
    if actos is not None:
        actos = actos_lista + actos          # la lista entra SIEMPRE al turno
    if not actos:
        # Extracción sin actos: puede ser fallo real O un simple saludo/charla.
        # En el turno de PRESENTACIÓN, "hola" no es un error: el bot se presenta
        # y ofrece el servicio, sin añadir "no te he entendido" detrás. Solo se
        # emite sin_entender cuando NO es intro (el cliente intentó algo que no
        # se entendió). Inv. 7 intacto: siempre sale un mensaje.
        intro = intro_mode in ("new", "renew")
        resultado = aplicar(estado_in, [], fuente_media=False, cantidad_global_pre=None)
        hechos = list(resultado.hechos)
        if not intro:
            hechos = [Hecho("sin_entender")] + hechos
        plantilla = render(hechos, resultado.estado, intro_mode=intro_mode,
                           client_name=client_name, salesman_name=salesman_name,
                           commercial_turn=intro)
        return _result(plantilla, resultado.estado, hechos,
                       {"degradado": "extraccion" if actos is None else "sin_actos"})

    # -------- Etapa 2
    async def _match_call(payload: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        data = await _chat_json(_MATCH_SYSTEM, _match_user(payload), MATCH_SCHEMA,
                                max_tokens=MATCH_MAX_TOKENS, timeout=MATCH_TIMEOUT_S,
                                temperature=0.0, budget=budget, kind="match", fmt="json")
        return (data or {}).get("items") if data else None

    res = await resolver(actos, estado_in, search=search, code_lookup=code_lookup,
                         match_call=_match_call,
                         cantidad_global_pre=pre.cantidad_global)

    # -------- Etapa 3
    resultado = aplicar(estado_in, res.ops, fuente_media=pre.fuente_media,
                        cantidad_global_pre=pre.cantidad_global)
    if resultado.descartes:
        logger.info("[PIPE] descartes: " + "; ".join(resultado.descartes))

    commercial = bool(resultado.estado.cart) or \
        resultado.estado.order_status in ("BUILDING", "CLOSED") or \
        any(h.tipo in ("pregunta_opciones", "sin_candidatos", "propuesta",
                       "conflicto_cantidad") for h in resultado.hechos)

    # -------- ¿callar? (§3.2) El bot contesta POCO. Calla cuando COMPRENDER
    # dijo hablar=false Y no hay nada del pedido que atender: o es charla no
    # comercial, o el cliente y el comercial ya están en una conversación
    # no-pedido (precios, detalles) y el bot no debe meterse en medio.
    # PERO: nunca calla ante algo del pedido, ni ante "pide_comercial" (avisa),
    # ni en el turno de presentación.
    NO_INTERPONERSE = {"cortesia", "derivacion"}   # charla o tema del comercial
    solo_no_pedido = resultado.hechos and all(h.tipo in NO_INTERPONERSE
                                              for h in resultado.hechos)
    pide_comercial = any(h.tipo == "aviso_comercial" for h in resultado.hechos)
    intro = intro_mode in ("new", "renew")
    if not hablar and solo_no_pedido and not pide_comercial and not intro:
        motivo = "tema_comercial" if any(h.tipo == "derivacion" for h in resultado.hechos) \
            else "charla"
        logger.info(f"[PIPE] silencio: no interponerse ({motivo})")
        return _result("", resultado.estado, [], {"silencio": motivo})

    # -------- Etapa 4: the template IS the reply. Built from the reducer's
    # facts and the cart, it cannot contradict them, so no verify step.
    mensaje = render(resultado.hechos, resultado.estado, intro_mode=intro_mode,
                     client_name=client_name, salesman_name=salesman_name,
                     commercial_turn=commercial)
    logger.info(f"[PIPE] turno ok: llamadas={budget.used} hechos="
                f"{[h.tipo for h in resultado.hechos]} cart={len(resultado.estado.cart)} "
                f"status={resultado.estado.order_status}")
    return _result(mensaje, resultado.estado, resultado.hechos)