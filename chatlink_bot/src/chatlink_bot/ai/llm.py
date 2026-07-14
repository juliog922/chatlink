# chatlink_bot/src/chatlink_bot/ai/llm.py
"""
Kapa, as ONE agent loop instead of two disconnected passes.

The old design ran Pass A (state updater that invented `search_queries`),
then RAG, then Pass B (reply writer). The model that wrote the queries never
saw the results, and the model that saw the results could not search again —
so casual client language ("la crema esa de siempre") produced one blind
query and a reply built on noise.

Here a single model call chain drives everything through tools:

    search_products  -> executed LIVE (retriever injected by the caller);
                        results are appended as a tool message and the model
                        is called again (up to MAX_STEPS), so it can refine
                        its query before answering. This is what implements
                        "nicely redirect the client to exact catalog names".
    add_item / remove_item / set_qty -> cart edits, validated IN CODE.
                        add_item only accepts codes the model has actually
                        seen (current cart or this turn's search results);
                        unseen codes are demoted to a search — hallucinated
                        codes are impossible, not merely discouraged.
    close_order      -> order_status = CLOSED (final confirmation).
    handoff_to_human -> caller sends the canned "the salesman will contact
                        you" message; no LLM text is delivered.
    opt_out_client   -> caller persists bot_enabled=False for this
                        conversation; the bot goes permanently silent there.
    note             -> one-line rolling summary kept across turns.
    (no tool calls + empty/NO_REPLY text) -> silence.

All state invariants live in `_ToolExecutor`; the model can make a wrong
edit but can never corrupt the cart, duplicate codes, or set invalid enums.
Everything is stdlib + the existing CimaClient.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from .cima_client import CimaClient, Message

logger = logging.getLogger("LLM")

CIMA_URL = os.getenv("CIMA_URL", "http://cima:8000")
CIMA_MODEL = os.getenv("CIMA_MODEL", "unsloth/gemma-4-E2B-it-GGUF:Q8_0")
CTX_WINDOW = int(os.getenv("CIMA_CTX_WINDOW", "8192"))
MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "3"))          # LLM calls per turn
MAX_HISTORY_CHARS = int(os.getenv("AGENT_HISTORY_MAX_CHARS", "4000"))
AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.55"))  # variety without breaking tool JSON

# ---- context budget -------------------------------------------------------
# The assembled prompt (system + state + history + message + tool results)
# must stay under AGENT_CTX_BUDGET_PCT of the model's window, leaving the rest
# for generation. When it doesn't fit, the sacrifices are ordered by value:
# history is trimmed tail-first (the persisted `note` summary — always in the
# prompt as "Memoria" — carries the older context), then oversized messages
# (huge PDF/image extractions) are middle-cut, then tool results are capped.
CTX_BUDGET_TOKENS = int(CTX_WINDOW * float(os.getenv("AGENT_CTX_BUDGET_PCT", "0.70")))
MAX_MESSAGE_CHARS = int(os.getenv("AGENT_MESSAGE_MAX_CHARS", "2500"))
MAX_RESULTS_CHARS = int(os.getenv("AGENT_RESULTS_MAX_CHARS", "3000"))
_GENERATION_SLACK_TOKENS = 96   # role markers, chat template overhead


def _est_tokens(text: str) -> int:
    """Conservative estimate (~3 chars/token for Spanish)."""
    return max(1, len(text) // 3)


def _cap_middle(text: str, max_chars: int) -> str:
    """Keep head and tail of oversized text (media extractions can be huge)."""
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.75)
    tail = max_chars - head
    return text[:head].rstrip() + "\n…[recortado]…\n" + text[-tail:].lstrip()


def _tail_lines_to_fit(history: str, max_chars: int) -> str:
    """Newest history lines that fit the budget; older context lives in Memoria."""
    if max_chars <= 0:
        return ""
    if len(history) <= max_chars:
        return history
    kept: List[str] = []
    used = 0
    for line in reversed(history.splitlines()):
        cost = len(line) + 1
        if used + cost > max_chars:
            break
        kept.append(line)
        used += cost
    return "\n".join(reversed(kept))

VALID_STATUS = ("IDLE", "BUILDING", "CLOSED")

# Retriever signature: queries -> {query: [{"CodigoArticulo", "DescripcionArticulo", ...}]}
Retriever = Callable[[List[str]], Awaitable[Dict[str, List[Dict[str, Any]]]]]

_client: Optional[CimaClient] = None


def _get_client() -> CimaClient:
    global _client
    if _client is None:
        _client = CimaClient(CIMA_URL)
    return _client


async def health_check() -> Dict[str, Any]:
    """cima reachability probe (used by /api/healthz)."""
    url = f"{CIMA_URL}/api/version"
    try:
        code = await asyncio.to_thread(
            lambda: urllib.request.urlopen(url, timeout=3).getcode()
        )
        return {"status": "ok" if code == 200 else "error", "code": code, "url": url}
    except Exception as e:
        return {"status": "error", "error": str(e), "url": url}


# --------------------------------------------------------------------- result
@dataclass
class AgentResult:
    """Everything handle_ai_trigger needs to act and persist. `reply == ""` means silence."""
    reply: str = ""
    order_status: str = "IDLE"
    cart: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    handoff: bool = False
    opt_out: bool = False
    # Multi-item work queue (unresolved products with their option snapshots):
    open_items: List[Dict[str, Any]] = field(default_factory=list)
    # One-time capability guidance already delivered in this conversation:
    guide_shown: bool = False
    # Context-window telemetry for the console gauge (est. tokens):
    ctx: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------- tools
_TOOLS: List[Dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "search_products",
        "description": ("Busca en el catálogo real. Códigos de producto (KG001399, 65012A…) "
                        "SIEMPRE como queries individuales y primero; después descripciones. "
                        "Incluye los códigos que aparezcan en [Texto en Imagen], [Audio transcrito] "
                        "o [Documento]. No busques lo que ya está en el carrito."),
        "parameters": {"type": "object", "properties": {
            "queries": {"type": "array", "items": {"type": "string"}}}, "required": ["queries"]},
    }},
    {"type": "function", "function": {
        "name": "add_item",
        "description": ("Añade un producto CONFIRMADO al carrito. SOLO cuando el cliente valida "
                        "explícitamente ('sí, ese', 'apúntame 2'). Buscar o mencionar NO es confirmar. "
                        "El código debe venir de los resultados de búsqueda."),
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string"}, "qty": {"type": "integer"}}, "required": ["code", "qty"]},
    }},
    {"type": "function", "function": {
        "name": "remove_item",
        "description": "Quita un producto del carrito cuando el cliente lo pide explícitamente.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
    }},
    {"type": "function", "function": {
        "name": "set_qty",
        "description": "Corrige la cantidad de un producto YA confirmado ('ponme 5 en vez de 3').",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string"}, "qty": {"type": "integer"}}, "required": ["code", "qty"]},
    }},
    {"type": "function", "function": {
        "name": "close_order",
        "description": "El cliente confirma el ENVÍO FINAL del pedido completo. Cierra el pedido.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "handoff_to_human",
        "description": ("El cliente pide hablar con el comercial, o pregunta por precios, stock, "
                        "promociones, facturación, incidencias, el estado/envío de un pedido ya "
                        "enviado, o pide consejo sobre productos (para qué sirve, cuál es mejor). "
                        "NUNCA la uses para preguntas sobre CÓMO hacer un pedido, sobre ti o "
                        "para identificar un producto del catálogo: eso lo respondes tú."),
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}},
    }},
    {"type": "function", "function": {
        "name": "opt_out_client",
        "description": ("El cliente expresa que NO quiere hablar con un asistente/bot. "
                        "Silencia el bot en esta conversación PARA SIEMPRE. Úsalo a la primera señal clara."),
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "note",
        "description": "Resumen brevísimo (1 frase) del estado de la conversación, para memoria interna.",
        "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
    }},
]

_SYSTEM = """\
QUIÉN ERES: Kapa, el asistente de pedidos de {salesman_name}, vendedor de \
cosmética. Atiendes a sus clientes por chat cuando él no está. NO eres \
{salesman_name} ni ves sus charlas: ante "el que me comentaste/comentó" sin \
rastro en tu historial, dilo natural — que lo confirme {salesman_name} o te \
den el nombre — y sigue con el resto. En el historial, "Cliente:" es el \
cliente y "Asistente:" eres TÚ: no confundas quién dijo qué ni repitas tus \
frases. Si ya te presentaste en el historial, no saludes ni te presentes.

TU META: construir el pedido identificando el CÓDIGO DE PRODUCTO exacto de \
cada artículo (dilo siempre así: "código de producto"). Todo lo demás es de \
{salesman_name}.

TU MUNDO: un catálogo de miles de productos, nombres en español/inglés \
mezclados, cada uno con marca y un código de producto único. El cliente casi \
nunca lo sabe y describe vago; el código de producto es la única \
identificación segura, tu búsqueda por nombre/palabras clave encuentra \
candidatos para elegir. Tú puedes leer mal audios/imágenes y las búsquedas \
pueden fallar: lo tuyo son interpretaciones, no certezas.

ESTILO: WhatsApp natural, frases cortas, algún emoji. Sigue el hilo, en el \
idioma del cliente. No menciones sistema/IA. Las [INSTRUCCIÓN INTERNA] no \
son del cliente: cúmplelas sin citarlas ni disculparte. {intro_rule}

BUSCAR (search_products):
- Solo si nombra producto, marca o código; sin términos genéricos ni \
ofrecer lo que nadie pidió.
- Consultas cortas (1-3 palabras, producto primero), nunca la frase entera. \
VARIOS productos: UNA consulta por producto. UNO y vago: 2-3 variantes \
("quiero la crema para la cara nivea" → ["crema nivea", "crema facial", \
"crema cara"]). Códigos ("14-1127") solos y primero. "Crema" a secas \
no se busca: enriquece con lo que sepas o pide detalle (código de producto = \
lo más exacto); si confirma que se llama así tal cual, búscalo.
- VARIOS PRODUCTOS: los resultados llegan triados con sus "instruccion_*": \
síguelas; UNA sola pregunta por turno. Tras resolver un pendiente, sigue \
con el siguiente de "Pendientes de identificar" (ya tienes sus opciones).
- El mensaje continúa el historial: hablabais de delineadores y "los \
quiero de Gelfix" → ["delineador gelfix", "gelfix"]. Una marca sola se busca; \
un color o atributo solo NUNCA: "el negro" tras Gelfix → ["gelfix negro"]. No preguntes "¿qué tipo?" si el historial lo dice.
- "la 2" → el código de esa línea de TU última lista.
- Nunca digas "un momento" / "voy a mirar": busca YA o responde con lo que sabes.

RESULTADOS:
- Muestra solo lo que encaje con lo pedido. Si no encaja (pide crema y salen \
esmaltes o carteles): dilo y pide marca, detalle o el código exacto.
- Si pide una MARCA y ningún resultado la tiene: di que no la ves en el catálogo.
- Propuestas: reacciona PRIMERO al mensaje con tus palabras — nunca repitas \
frases de mensajes anteriores — lista numerada real y pregunta final \
variada. Ejemplo:
  "¡De Gelfix sí que tengo! 👇
   1) EJ0001 — CREMA FACIAL HIDRATANTE (ANUBIS)
   ¿Cuál te apunto?"
  Varía aperturas ("Mira lo que encontré:", "Estas te pueden valer:") y \
cierres ("¿cuál prefieres?", "dime número o código", "si no era esto, dame \
marca o código y lo afino").
- HUMILDAD: puedes fallar leyendo imágenes/audio o buscando. \
Con tus palabras (sin frase fija), di que ofreces lo que TÚ entendiste e \
invita a corregirte con detalle o código de producto. UNA mención basta; sin \
disculpas continuas.

PEDIDO:
- Solo existe lo que devuelven las búsquedas: PROHIBIDO inventar códigos, \
nombres o marcas.
- add_item con confirmación ("sí, ese") o cuando el triaje diga "apúntalos"; \
buscar NO es confirmar. Todo revisable hasta el cierre.
- El carrito se mantiene entre mensajes; solo cambia si el cliente añade, \
quita o corrige. Resumen: "• 2× EJ0001 — CREMA FACIAL" (reales, nunca \
"CANT"; los EJ000x de ejemplo NO existen).
- Carrito lleno y sin novedades: "¿Añadimos algo más o te cierro el pedido?"
- "nada más" / "eso es todo" / "ya está" / "envíalo" → resumen completo + \
close_order + "Le paso la nota a {salesman_name}. ¡Gracias {client_name}!"
- Pedido YA enviado no se toca: pedido nuevo o handoff_to_human. \
"Lo de siempre" → último pedido enviado del contexto.

SI PREGUNTAN CÓMO PEDIR (sin buscar): dime productos (nombre o, mejor, \
código exacto) y cantidades; los confirmo; con tu "sí" los apunto; al cerrar \
le paso la nota a {salesman_name}. Sin enseñar catálogo ni recomendar.

DERIVA a handoff_to_human: precios/descuentos, stock, facturas/pagos, \
incidencias/devoluciones, envío o estado de un pedido enviado, consejos de producto. Si MEZCLA pedido \
y precio/duda: sigue el pedido y di tú, breve, que eso se lo confirmará \
{salesman_name}; handoff solo si no hay pedido que atender. Puedes dar el NOMBRE del catálogo para \
identificar el código; nunca describir ni recomendar.
Rechaza hablar con asistentes → opt_out_client. Al final, note (1 frase, invisible al cliente).
Mensaje ajeno al negocio sin nada útil para el pedido → "<NO_REPLY>".
[Texto en Imagen]/[Audio transcrito]/[Documento] → demuestra que lo leíste; \
puedes haberlo leído mal. Marcador "(...no entendible/legible/\
procesable...)": dilo breve, sin dramatismo; pide texto u otro formato. \
De audio/imagen NUNCA apuntes NI \
cierres directo: propón lo entendido y espera el "sí". "(transcripción dudosa)": \
confirma antes de buscar."""

_INTRO_RULES: Dict[str, str] = {
    "new": ("CONTACTO NUEVO: preséntate en UNA frase (Kapa, asistente de {salesman_name}) y "
            "EN LA MISMA respuesta atiende lo que haya pedido (busca/propón). Solo si "
            "únicamente saluda, pregúntale con tus palabras qué necesita."),
    "renew": ("La conversación anterior TERMINÓ. Si empieza una gestión de pedido, preséntate "
              "breve de nuevo antes de atenderle; si solo saluda, agradece o se despide, "
              "responde cordial SIN presentarte (o <NO_REPLY> si no aporta nada)."),
    "ongoing": ("Conversación en curso: PROHIBIDO presentarte o saludar de nuevo; "
                "responde directo al mensaje."),
}

_USER = """\
### CONTEXTO
- Estado del pedido: {order_status}
- Pendientes de identificar: {open_items_line}
- Carrito confirmado: {cart_json}
- Último pedido ENVIADO (no modificable): {last_closed_json}
- Memoria: {summary}
- Historial reciente:
{history}

### MENSAJE(S) NUEVO(S) DEL CLIENTE
{current_message}

Actualiza el estado con las herramientas que hagan falta y escribe tu respuesta \
final para el cliente (o <NO_REPLY>)."""


# ------------------------------------------------------------------- executor
# Queries where EVERY token is a generic catalog word carry no product intent
# ("productos", "el catálogo entero"...). Observed live: the model searched
# 'productos' when asked HOW to order, and RAG dutifully returned three random
# items that Kapa then pushed on the client. The prompt forbids it, but a
# 2B model needs the rule enforced in code: these queries are dropped before
# any search runs. Specific queries ("crema kerapro", "14-1127") pass.
_GENERIC_QUERY_TOKENS = frozenset(
    "producto productos articulo articulos artículo artículos catalogo catálogo "
    "pedido pedidos orden lista listado cosas algo todo todos disponible "
    "disponibles oferta ofertas novedades novedad opciones muestras stock "
    "precio precios comprar pedir tienes tiene hay que qué los las unos unas "
    "del de el la un una tu tus mis para".split()
)
_re_query_tokens = re.compile(r"[^\w]+", re.UNICODE)


def _is_generic_query(query: str) -> bool:
    tokens = [t for t in _re_query_tokens.split(query.lower()) if t]
    return bool(tokens) and all(t in _GENERIC_QUERY_TOKENS or len(t) < 3 for t in tokens)


# Escalation tools are honored immediately ONLY when the client's own message
# corroborates them — observed live: "¿cómo hago un pedido?" triggered
# handoff_to_human because 'pedido' smells like salesman territory to a 2B
# model, bouncing a trivial process question to a human. Uncorroborated calls
# are demoted once with a nudge to answer directly; if the model INSISTS on a
# second call, it is honored (covers real cases no keyword list can foresee).
_HANDOFF_SIGNAL_WORDS = frozenset(
    "precio precios cuesta cuestan vale valen coste costo tarifa descuento "
    "oferta ofertas promocion promociones rebaja factura facturas facturacion "
    "pago pagos abono stock disponibilidad incidencia incidencias problema "
    "problemas queja quejas reclamacion reclamaciones devolucion devoluciones "
    "devolver roto rota defectuoso defectuosa retraso urgente hablar llamar "
    "llamada llamame telefono contacto humano persona comercial vendedor "
    # Order status / delivery of a dispatched order -> salesman territory.
    "envio envios entrega entregas llega llegara llegado enviado enviaron "
    "seguimiento tracking transporte agencia paquete "
    # Product advice / characteristics -> salesman territory (Kapa identifies
    # codes, it does not consult).
    "sirve sirven recomienda recomiendas recomendacion ingredientes "
    "propiedades funciona composicion caducidad aplica".split()
)
_OPTOUT_SIGNAL_WORDS = frozenset(
    "bot robot maquina asistente automatico automatica ia humano persona "
    "molestes escribas contigo".split()
)


def _strip_accents_lower(s: str) -> str:
    import unicodedata
    s = (s or "").lower()
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def _message_has_signals(message: str, signals: frozenset, extra_words: str = "") -> bool:
    tokens = set(_re_query_tokens.split(_strip_accents_lower(message)))
    extra = set(_re_query_tokens.split(_strip_accents_lower(extra_words)))
    return bool(tokens & signals) or bool(extra and tokens & extra)


# A reply that ANNOUNCES a search without calling the tool leaves the client
# waiting forever — observed live: "Busca un momento, reviso qué tengo" with
# zero tool calls. Detected in code and demoted with a nudge to actually call
# search_products (narrating an action instead of performing it is a classic
# small-model failure that prompt rules alone don't eliminate).
_re_search_promise = re.compile(
    r"(un moment|un segund|voy a buscar|voy a mirar|voy a revisar|dejame (ver|buscar|mirar|revisar)"
    r"|revis[oa]\b|enseguida te|ahora (te )?(digo|miro|busco|reviso)|dame un)"
)


def _promises_search(reply: str) -> bool:
    return bool(_re_search_promise.search(_strip_accents_lower(reply)))


# One worked example in the prompt = a template a 2B model parrots verbatim —
# observed live: three consecutive proposals with the identical skeleton and
# zero reaction to the client's words. A PROPOSAL reply (>=2 numbered lines)
# that reuses one of its framing sentences verbatim from a previous bot
# message in the history is demoted once and rewritten.
_re_numbered_line = re.compile(r"^\s*\d+\)")

# Ungrounded proposals: the reply TEXT can name any code — from Memoria, the
# last closed order, old history, or even the prompt's own worked examples —
# and present it as "found". Observed live: a black-eyeliner search returned 3
# products and the reply added 2 Gelfix items nobody asked about. Proposals
# may only contain codes from THIS turn's search results (plus the cart; plus
# the last closed order when the client signals repetition). One rewrite
# nudge; then offending lines are stripped and the list renumbered.
_REPEAT_SIGNAL_WORDS = frozenset(
    "siempre mismo misma mismos mismas ultimo ultima anterior repetir "
    "repite repiteme habitual".split()
)
_re_item = re.compile(r"\d+\)\s*([A-Za-z0-9][A-Za-z0-9\-./]{1,})\s*[—-]")
_re_item_span = re.compile(r"\d+\)\s*([A-Za-z0-9][A-Za-z0-9\-./]{1,})\s*[—-][^\n]*?(?=\s*\d+\)|$)")
_re_item_marker = re.compile(r"\d+\)")


def _proposal_codes(reply: str) -> List[str]:
    """Codes offered as catalog items ('N) CODE — NAME'), including proposals
    flattened onto one line. The 'CODE —' shape keeps narrative enumerations
    ('1) Me dices los productos...') out of scope."""
    return [_sane_code(c) for c in _re_item.findall(reply or "")]


def _strip_ungrounded_lines(reply: str, bad_codes: set) -> str:
    """Remove ungrounded item spans (whole line if it only carries bad items,
    surgical within mixed lines), then renumber all items sequentially."""
    kept: List[str] = []
    for line in reply.splitlines():
        codes = {_sane_code(c) for c in _re_item.findall(line)}
        if codes and codes <= bad_codes:
            continue
        if codes & bad_codes:
            line = _re_item_span.sub(
                lambda m: "" if _sane_code(m.group(1)) in bad_codes else m.group(0), line).rstrip()
        kept.append(line)
    text = "\n".join(kept)
    counter = [0]

    def renum(m):
        counter[0] += 1
        return f"{counter[0]})"
    return _re_item_marker.sub(renum, text).strip()


def _is_parroting(reply: str, recent_history: str) -> bool:
    lines = reply.splitlines()
    is_proposal = sum(1 for l in lines if _re_numbered_line.match(l)) >= 2
    previous_bot_text = " | ".join(
        re.sub(r"\s+", " ", _strip_accents_lower(l.split(":", 1)[1]))
        for l in recent_history.splitlines() if l.startswith("Asistente:")
    )
    if not previous_bot_text:
        return False
    for line in lines:
        if _re_numbered_line.match(line):
            continue
        normalized = re.sub(r"\s+", " ", _strip_accents_lower(line)).strip()
        if is_proposal and len(normalized) >= 12 and normalized in previous_bot_text:
            return True
    if not is_proposal:
        # Non-proposal replies (e.g. the how-it-works talk) parrot when they
        # replay >= 2 long verbatim sentences from prior bot turns; a single
        # reused formula ("¿Añadimos algo más...?") stays exempt.
        sents = [re.sub(r"\s+", " ", _strip_accents_lower(s)).strip()
                 for s in re.split(r"[.!?\n]+", reply)]
        return sum(1 for s in sents if len(s) >= 25 and s in previous_bot_text) >= 2
    return False


# A bare ATTRIBUTE (color, size, finish, variant number) identifies nothing in
# an 8k-product catalog — observed live: after three messages about Gelfix
# eyeliners, "El negro" was searched as 'negro' and returned black files and
# clips. The current product family is derivable deterministically from the
# bot's own LAST numbered proposal in the history (tokens repeated across its
# items, e.g. KATAI/GELFIX), so attribute-only queries are grounded in code:
# 'negro' -> 'katai gelfix negro'. No topic available -> the query is dropped.
_ATTRIBUTE_TOKENS = frozenset(
    "negro negra blanco blanca rojo roja rosa rosado azul verde amarillo "
    "morado violeta lila dorado dorada plateado plateada gris marron beige "
    "nude transparente incoloro mate brillo brillante satinado metalizado "
    "claro clara oscuro oscura grande pequeno pequena mini midi maxi fino "
    "fina grueso gruesa corto corta largo larga suave fuerte".split()
)
_re_proposal_item = re.compile(r"\d+\)\s*[A-Za-z0-9\-./]+\s*—\s*([^0-9)][^)]{2,80}?)(?=\s+\d+\)|$)")


def _is_attribute_only(query: str) -> bool:
    tokens = [t for t in _re_query_tokens.split(_strip_accents_lower(query)) if t]
    return bool(tokens) and all(
        t in _ATTRIBUTE_TOKENS or (t.isdigit() and len(t) <= 4) or len(t) < 3
        for t in tokens
    )


def _topic_from_history(recent_history: str) -> List[str]:
    """Product-family tokens from the newest bot proposal in the history
    (>=2 numbered items: tokens repeated across item names). No proposal =
    no established product context — attribute-only queries then get dropped
    and the model asks which product the attribute refers to."""
    from collections import Counter
    for line in reversed((recent_history or "").splitlines()):
        if not line.startswith("Asistente:"):
            continue
        names = _re_proposal_item.findall(line)
        if len(names) < 2:
            continue
        counts: Counter = Counter()
        for name in names:
            counts.update({
                t for t in _re_query_tokens.split(_strip_accents_lower(name))
                if len(t) >= 4 and t not in _GENERIC_QUERY_TOKENS and t not in _ATTRIBUTE_TOKENS
            })
        common = [t for t, c in counts.most_common(4) if c >= 2][:2]
        if common:
            return common
    return []


# The `note` tool sometimes leaks as visible text ("Nota: el cliente quiere
# hacer un pedido") instead of a tool call. Scrub such trailing lines from the
# client-facing reply and reroute them to the session summary.
_re_leaked_note = re.compile(r"^\s*\(?\s*(nota|nota interna|memoria|resumen)\s*[:\-]\s*(.+?)\)?\s*$",
                             re.IGNORECASE)


def _split_leaked_note(reply: str) -> Tuple[str, str]:
    kept: List[str] = []
    note = ""
    for line in (reply or "").splitlines():
        m = _re_leaked_note.match(line)
        if m:
            note = m.group(2).strip()
            continue
        kept.append(line)
    return "\n".join(kept).strip(), note


# Humility guard (media turns): a proposal built from an image/audio/document
# interpretation must acknowledge fallibility and invite correction (keywords,
# brand, or the exact code). Enforced softly — the SIGNALS are checked in
# code, the WORDING stays the model's own, so no fixed phrase can calcify
# into a template. One nudge; never fires on plain-text turns.
_CORRECTION_SIGNALS = (
    "codigo", "corrig", "equivoc", "no era", "era esto", "confirmam", "confirma",
    "interpretad", "entendido", "he entendido", "he leido", "leido bien",
    "si no es", "no es lo que", "afino", "asegurar", "me falta algo",
)
_MEDIA_MARKERS = ("[Texto en Imagen]", "[Audio transcrito]", "[Documento")


def _invites_correction(reply: str) -> bool:
    norm = _strip_accents_lower(reply)
    return any(s in norm for s in _CORRECTION_SIGNALS)


def _is_media_message(message: str) -> bool:
    return any(m in (message or "") for m in _MEDIA_MARKERS)


# ------------------------------------------------------- searchability gate
# "crema" alone against a catalog with hundreds of creams is not a search,
# it is noise: Qdrant will return three arbitrary creams and the agent will
# play fortune teller with them. The gate holds such queries BEFORE any
# retrieval. Recovery ladder (cheapest first):
#   1. the model self-enriches from message/history context (one nudge),
#   2. the client is asked for name/keywords/code (item queued as
#      estado="enriquecer" so it is never lost),
#   3. the client insists the product is really called just that ->
#      the insistence bypass lets that exact query through once.
_GENERIC_NOUNS = {
    "crema", "cream", "esmalte", "polish", "gel", "base", "champu", "shampoo",
    "jabon", "soap", "aceite", "oil", "polvo", "powder", "brocha", "pincel",
    "brush", "lima", "file", "mascarilla", "mask", "laca", "serum", "spray",
    "kit", "pack", "producto", "product", "articulo", "cosmetico", "top",
    "esponja", "sponge", "toalla", "towel", "locion", "lotion", "tinte",
}
_QUERY_STOP = {"de", "la", "el", "los", "las", "un", "una", "unos", "unas",
               "para", "por", "con", "y", "o", "del", "al", "en", "mi", "tu"}


def _query_tokens(q: str) -> List[str]:
    norm = _strip_accents_lower(q)
    return [t for t in re.split(r"[^a-z0-9ñ]+", norm)
            if len(t) >= 3 and t not in _QUERY_STOP and not t.isdigit()]


def _code_shaped(q: str) -> bool:
    q = (q or "").strip()
    return len(q) >= 4 and " " not in q and any(ch.isdigit() for ch in q)


def _too_generic(q: str) -> bool:
    """True when the query cannot reasonably identify a product: no usable
    tokens, or a single bare generic noun. Code-shaped queries always pass."""
    if _code_shaped(q):
        return False    # code-shaped: the exact-code path will judge it
    toks = _query_tokens(q)
    if not toks:
        return True
    return len(toks) == 1 and toks[0] in _GENERIC_NOUNS


# ---------------------------------------------------------- multi-item triage
# A client listing several products must be handled like a salesman with a
# catalog: items whose top hit is unambiguous get confirmed in ONE line each;
# exactly ONE doubtful item gets its options asked about; the rest wait in an
# explicit queue that persists across turns (open_items). Without this, an
# 8-product message produced a 24-line hit dump — technically complete,
# humanly useless. The LLM only phrases the digest; code decides it.

def _query_verdict(hits: List[Dict[str, Any]]) -> str:
    """resolved -> note it; leading -> pick it and mention the alternative in
    passing; ambiguous -> the ONE question this turn; none -> ask for detail.
    The 'leading' tier is what keeps a long list from feeling like an
    interrogation: a salesman assumes the obvious and offers an escape hatch
    ("te pongo el de 100ml; si era el litro, dime") instead of asking."""
    if not hits:
        return "none"
    if len(hits) == 1:
        return "resolved"
    full = [h for h in hits if h.get("covers_query")]
    if len(full) == 1:
        return "resolved"       # one candidate matches every word the client used
    s0 = float(hits[0].get("relevance_score", 0.0))
    s1 = float(hits[1].get("relevance_score", 0.0))
    if s0 >= 2.0:
        return "resolved"       # exact-code short-circuit
    if s1 > 0 and s0 / s1 >= 1.5:
        return "leading"        # clear favorite: assume it, mention the runner-up
    return "ambiguous"


def _hit_line(h: Dict[str, Any]) -> Dict[str, str]:
    return {"codigo": str(h.get("CodigoArticulo") or ""),
            "nombre": str(h.get("DescripcionArticulo") or "")[:80],
            "marca": str(h.get("MarcaProducto") or "")}


def _merge_variant_queries(results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """Query VARIANTS of the same item ('crema nivea'/'crema facial') return
    overlapping hits; merge them so triage counts items, not queries."""
    merged: Dict[str, List[Dict[str, Any]]] = {}
    code_sets: Dict[str, set] = {}
    for q, hits in results.items():
        codes = {str(h.get("CodigoArticulo")) for h in hits}
        top = str(hits[0].get("CodigoArticulo")) if hits else None
        target = next((k for k, cs in code_sets.items() if codes and cs and (
            (top is not None and merged[k] and str(merged[k][0].get("CodigoArticulo")) == top)
            or len(codes & cs) / len(codes | cs) >= 1 / 3)), None)
        if target is None:
            merged[q] = list(hits)
            code_sets[q] = codes
        else:
            seen = {str(h.get("CodigoArticulo")) for h in merged[target]}
            merged[target].extend(h for h in hits if str(h.get("CodigoArticulo")) not in seen)
            code_sets[target] |= codes
    return merged


def _triage_results(results: Dict[str, List[Dict[str, Any]]],
                    carried_queue: List[Dict[str, Any]],
                    is_media: bool = False) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """-> (digest for the model, open_items to persist).

    Digest keys are Spanish on purpose: the 2B model relays them naturally.
    The instruccion_* strings are CONTEXTUAL guidance — they cost tokens only
    on the turns that need them, keeping the static system prompt small.

    Turn economy (the anti-interrogatory design):
    - resolved  -> add NOW, confirm in one line; the close summary is the one
                   explicit confirmation gate (everything revisable until then)
    - leading   -> add the favorite, mention the runner-up as an escape hatch
    - ambiguous -> at most ONE question per turn; the rest wait in the queue
    - media turns switch every 'add' instruction to propose-and-confirm so
      this never fights the hallucination containment."""
    merged = _merge_variant_queries(results)
    resolved, chosen, ambiguous, missing = [], [], [], []
    honesty_needed = False
    for q, hits in merged.items():
        verdict = _query_verdict(hits)
        uncovered = list(hits[0].get("query_uncovered") or []) if hits else []
        if uncovered and verdict in ("resolved", "leading"):
            # Something the client SAID ('nivea') is in none of the options:
            # confidence would be fortune-telling. Demote to an honest ask —
            # never auto-add a silent substitute.
            verdict = "ambiguous"
        if verdict == "resolved":
            resolved.append({"pedido": q, **_hit_line(hits[0])})
        elif verdict == "leading":
            alt = _hit_line(hits[1])
            chosen.append({"pedido": q, **_hit_line(hits[0]),
                           "alternativa": f"{alt['codigo']} — {alt['nombre']}"})
        elif verdict == "none":
            missing.append(q)
        else:
            item: Dict[str, Any] = {"pedido": q, "opciones": [_hit_line(h) for h in hits[:3]]}
            if uncovered:
                item["sin_coincidencia"] = uncovered
                honesty_needed = True
            ambiguous.append(item)
    # Previously queued items go first: the client is mid-list.
    queue = list(carried_queue) + ambiguous
    ask_now = queue[0] if queue else None
    remaining = queue[1:]
    digest: Dict[str, Any] = {}
    n_items = len(resolved) + len(chosen) + len(queue) + len(missing)
    if n_items > 1:
        digest["instruccion_general"] = (
            "empieza reconociendo el pedido COMPLETO en una frase natural (qué tienes "
            "claro, qué tiene opciones, a qué le faltan datos) y luego desarrolla; "
            "una sola pregunta")
    if resolved:
        digest["identificados_seguros"] = resolved
        digest["instruccion_identificados"] = (
            "propónlos en una línea cada uno y espera el sí del cliente" if is_media else
            "apúntalos YA con add_item (cantidad que dijo el cliente; si no dijo, 1) y "
            "dilo en una línea cada uno; todo es revisable hasta el cierre")
    if chosen:
        digest["elegidos_probables"] = chosen
        digest["instruccion_elegidos"] = (
            "propón el elegido mencionando la alternativa y espera el sí" if is_media else
            "apúntalos YA con add_item y di de pasada que has elegido ese "
            "(ej.: 'te pongo X; si preferías la alternativa, me dices')")
    if ask_now:
        digest["pregunta_ahora"] = ask_now
        digest["instruccion_pregunta"] = ("ofrece SOLO estas opciones numeradas, pide que elija "
                                          "y varía la forma de preguntar; si dice que ninguna "
                                          "vale, pide nombre exacto o código de producto, apúntalo en note "
                                          "y NO vuelvas a ofrecer esas")
    if remaining:
        digest["en_cola"] = [i["pedido"] for i in remaining]
        digest["instruccion_cola"] = ("di brevemente que los verás uno a uno" +
                                      ("; si te da códigos o marcas de esos, mejor"
                                       if len(remaining) > 1 else ""))
    if honesty_needed:
        digest["instruccion_honestidad"] = (
            'si un artículo trae "sin_coincidencia", di CLARO que ninguna opción tiene eso; '
            "que elija una igualmente, dé el código de producto, o se lo deja al comercial")
    if missing:
        digest["sin_resultados"] = missing
        digest["instruccion_sin_resultados"] = "pide marca, más detalle o el código de producto"
    return digest, queue    # asked item stays queued until the client resolves it


def _internal(text: str) -> str:
    """Wrap loop nudges: instructions, not client messages. The system prompt
    teaches the model to follow these without quoting or apologizing."""
    return f"[INSTRUCCIÓN INTERNA — el cliente NO ha escrito esto] {text}"


def _sane_code(code: Any) -> str:
    return str(code or "").strip()[:64]


def _sane_qty(qty: Any) -> int:
    try:
        return max(1, min(int(qty), 9999))
    except Exception:
        return 1


class _ToolExecutor:
    """Applies tool calls to the state, enforcing every invariant in code."""

    def __init__(self, session: Dict[str, Any], topic: Optional[List[str]] = None,
                 media_text: str = "") -> None:
        self.topic: List[str] = topic or []
        # Media turns (image/audio/doc transcriptions) may be machine
        # misreadings: adds are allowed ONLY for codes literally present in
        # the extraction (dictated codes); anything else must be proposed
        # and confirmed. Alphanumeric-normalized containment check.
        self._media_alnum: Optional[str] = (
            re.sub(r"[^A-Z0-9]", "", media_text.upper()) if media_text else None)
        self.blocked_media_adds: List[str] = []
        self.blocked_media_close: bool = False
        # Multi-item work queue persisted across turns; option codes are
        # catalog-verified from a prior search -> grounded for proposals/adds.
        self.open_items: List[Dict[str, Any]] = list(session.get("open_items") or [])
        self._guide_shown_out: bool = bool(session.get("guide_shown"))
        self.held_for_enrichment: List[str] = []
        self.blocked_empty_close: bool = False
        self.order_activity: bool = False   # any cart-directed action this turn
        self.results_this_turn: set = set()
        self.cart: Dict[str, Dict[str, Any]] = {
            i["code"]: {"code": i["code"], "qty": _sane_qty(i.get("qty"))}
            for i in (session.get("cart") or []) if i.get("code")
        }
        self.status: str = session.get("order_status") or "IDLE"
        self.summary: str = session.get("summary") or ""
        self.summary_touched: bool = False
        # Grounding set for add_item: the current cart plus the last DISPATCHED
        # order — those codes were catalog-validated when first added, so
        # "ponme lo mismo que la última vez" works without a redundant search.
        self.seen_codes: set[str] = set(self.cart) | {
            _sane_code(o.get("codigo")) for item in (session.get("open_items") or [])
            for o in item.get("opciones", []) if o.get("codigo")
        } | {
            _sane_code(i.get("code")) for i in (session.get("last_closed_cart") or []) if i.get("code")
        }
        self.pending_adds: Dict[str, int] = {}        # rejected adds awaiting grounding
        self.handoff = False
        self.opt_out = False

    def apply(self, calls: List[Dict[str, Any]]) -> List[str]:
        """Apply one round of calls; return the search queries requested (deduped, grounded)."""
        queries: List[str] = []
        for call in calls:
            fn = call.get("function") or {}
            name, args = fn.get("name") or "", fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            if name == "search_products":
                for q in (args.get("queries") or []):
                    q = str(q or "").strip()
                    if not q or q in queries:
                        continue
                    if _is_generic_query(q):
                        logger.info(f"[AGENT] Dropped generic search query: '{q}'")
                        continue
                    if _is_attribute_only(q):
                        if not self.topic:
                            logger.info(f"[AGENT] Dropped bare attribute query (no topic): '{q}'")
                            continue
                        grounded = f"{' '.join(self.topic)} {q}"
                        logger.info(f"[AGENT] Attribute query grounded with topic: '{q}' -> '{grounded}'")
                        q = grounded
                    queries.append(q)
            elif name == "add_item":
                code = _sane_code(args.get("code"))
                if code and self._media_alnum is not None and \
                        re.sub(r"[^A-Z0-9]", "", code.upper()) not in self._media_alnum:
                    # Machine-read message that does NOT contain this code: the
                    # model is acting on history/imagination, not on what the
                    # client actually sent. Block; propose-and-confirm instead.
                    logger.warning(f"[AGENT] add_item BLOCKED on media turn "
                                   f"(code {code} not present in the extraction).")
                    self.order_activity = True
                    self.blocked_media_adds.append(code)
                elif code in self.seen_codes:
                    self.order_activity = True
                    self.cart[code] = {"code": code, "qty": _sane_qty(args.get("qty"))}
                    self.status = "BUILDING" if self.status != "CLOSED" else self.status
                    logger.info(f"[AGENT] add_item applied: {code} x{self.cart[code]['qty']} "
                                f"(cart={len(self.cart)})")
                    # Adding one of a queued item's options resolves that item.
                    self.open_items = [
                        i for i in self.open_items
                        if code not in {_sane_code(o.get("codigo")) for o in i.get("opciones", [])}
                    ]
                elif code:
                    # Unseen code (e.g. proposed in a PREVIOUS turn, or typed by the
                    # client): don't trust it — demote to a search. The exact-code
                    # short-circuit in RAG grounds real codes, and register_results()
                    # replays the add automatically once grounded, so a legitimate
                    # confirmation is never lost while invented codes stay impossible.
                    logger.info(f"[AGENT] add_item deferred for unseen code {code}; grounding via search.")
                    self.order_activity = True
                    self.pending_adds[code] = _sane_qty(args.get("qty"))
                    queries.append(code)
            elif name == "remove_item":
                self.cart.pop(_sane_code(args.get("code")), None)
            elif name == "set_qty":
                code = _sane_code(args.get("code"))
                if code in self.cart:
                    self.cart[code]["qty"] = _sane_qty(args.get("qty"))
            elif name == "close_order":
                if self._media_alnum is not None:
                    # A transcription may not command the MOST irreversible
                    # action — observed live: "…y cierra el pedido" by voice
                    # closed and SHIPPED a stale 1-item cart while the adds in
                    # the same audio sat blocked awaiting confirmation.
                    self.blocked_media_close = True
                    logger.warning("[AGENT] close_order BLOCKED on media turn "
                                   "(final confirmation must arrive as text).")
                elif not self.cart:
                    self.blocked_empty_close = True
                    logger.warning("[AGENT] close_order ignored: empty cart.")
                else:
                    self.status = "CLOSED"
            elif name == "handoff_to_human":
                self.handoff = True
            elif name == "opt_out_client":
                self.opt_out = True
            elif name == "note":
                s = str(args.get("summary") or "").strip()
                if s:
                    self.summary = s[:300]
                    self.summary_touched = True
            else:
                logger.debug(f"[AGENT] Ignoring unknown tool: {name}")
        if queries and self.status == "IDLE":
            self.status = "BUILDING"
        return queries

    def register_results(self, results: Dict[str, List[Dict[str, Any]]]) -> None:
        """Ground the codes returned by search, then replay any deferred adds."""
        for hits in results.values():
            codes = {_sane_code(h.get("CodigoArticulo")) for h in hits if h.get("CodigoArticulo")}
            self.seen_codes |= codes
            self.results_this_turn |= codes
        for code, qty in list(self.pending_adds.items()):
            grounded = next((c for c in self.seen_codes if c.upper() == code.upper()), None)
            if grounded:
                self.cart[grounded] = {"code": grounded, "qty": qty}
                if self.status != "CLOSED":
                    self.status = "BUILDING"
                del self.pending_adds[code]
                logger.info(f"[AGENT] Deferred add grounded and applied: {grounded} x{qty}")

    def result(self, reply: str) -> AgentResult:
        return AgentResult(
            reply=reply, order_status=self.status if self.status in VALID_STATUS else "IDLE",
            cart=list(self.cart.values()), summary=self.summary,
            open_items=self.open_items,
            guide_shown=bool(self._guide_shown_out),
            handoff=self.handoff, opt_out=self.opt_out,
        )


def _clean(text: str) -> str:
    """Strip reasoning tags and the silence token from model output."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    return "" if "<NO_REPLY>" in text else text


# ----------------------------------------------------------------------- loop
async def run_agent(
    *,
    client_name: str,
    salesman_name: str,
    session: Dict[str, Any],
    recent_history: str,
    current_message: str,
    intro_mode: str,          # "new" | "renew" | "ongoing"  (decided by handlers, in code)
    search: Retriever,
) -> AgentResult:
    """
    One conversation turn: LLM <-> tools loop (max MAX_STEPS calls), grounded
    live searches, code-validated state edits. Blocking cima calls run in a
    thread so the event loop stays free.
    """
    executor = _ToolExecutor(session, topic=_topic_from_history(recent_history),
                             media_text=current_message if _is_media_message(current_message) else "")
    intro_rule_text = _INTRO_RULES.get(intro_mode, _INTRO_RULES["ongoing"])
    if not session.get("guide_shown"):
        # Once per conversation: set expectations like a person would, then
        # never repeat it. Persisted via AgentResult.guide_shown.
        intro_rule_text += (
            " ADEMÁS (solo esta vez): explica en 1-2 frases naturales cómo trabajas mejor: "
            "el código de producto es lo más fácil (lo verificas al momento); si no, nombre casi exacto o "
            "palabras clave útiles y le das opciones; y lo que no salga, se lo deja a "
            "{salesman_name}.")
    intro = intro_rule_text.format(
        client_name=client_name, salesman_name=salesman_name)
    system = _SYSTEM.format(salesman_name=salesman_name, client_name=client_name, intro_rule=intro)

    # Budget-aware assembly: cap the message, then give history whatever
    # tokens remain under CTX_BUDGET_TOKENS. Older context lives in Memoria.
    current_message = _cap_middle(current_message or "", MAX_MESSAGE_CHARS)
    open_line = "; ".join(
        (f"{i.get('pedido')} (falta detalle del cliente)"
         if i.get("estado") == "enriquecer" else
         f"{i.get('pedido')} (opciones: " + ", ".join(
             o.get("codigo", "") for o in i.get("opciones", [])[:3]) + ")")
        for i in executor.open_items) or "(ninguno)"
    shell = _USER.format(
        order_status=executor.status,
        open_items_line=open_line,
        cart_json=json.dumps(list(executor.cart.values()), ensure_ascii=False),
        last_closed_json=json.dumps(session.get("last_closed_cart") or [], ensure_ascii=False),
        summary=executor.summary or "(sin memoria previa)",
        history="",
        current_message=current_message or "(sin texto)",
    )
    avail_tokens = CTX_BUDGET_TOKENS - _est_tokens(system) - _est_tokens(shell) - _GENERATION_SLACK_TOKENS
    history = _tail_lines_to_fit(recent_history or "", min(max(avail_tokens, 0) * 3, MAX_HISTORY_CHARS))

    messages: List[Message] = [
        Message(role="system", content=system),
        Message(role="user", content=_USER.format(
            order_status=executor.status,
            open_items_line=open_line,
            cart_json=json.dumps(list(executor.cart.values()), ensure_ascii=False),
            last_closed_json=json.dumps(session.get("last_closed_cart") or [], ensure_ascii=False),
            summary=executor.summary or "(sin memoria previa)",
            history=history or "(sin historial)",
            current_message=current_message or "(sin texto)",
        )),
    ]

    # ---- context telemetry (est. tokens; the gauge the console monitors) ----
    ctx_stats: Dict[str, Any] = {
        "window": CTX_WINDOW, "budget": CTX_BUDGET_TOKENS, "used": 0,
        "history_chars_kept": len(history), "history_chars_total": len(recent_history or ""),
        "message_truncated": "[recortado]" in (current_message or ""),
    }

    def _res(rep: str) -> AgentResult:
        r = executor.result(rep)
        used = ctx_stats["used"]
        r.ctx = {**ctx_stats,
                 "pct_window": round(100 * used / max(1, CTX_WINDOW), 1),
                 "pct_budget": round(100 * used / max(1, CTX_BUDGET_TOKENS), 1),
                 "history_trimmed": ctx_stats["history_chars_kept"] < ctx_stats["history_chars_total"]}
        return r

    async def _finish(rep: str) -> AgentResult:
        if rep.strip() and "<NO_REPLY>" not in rep:
            executor._guide_shown_out = True
            # Greeting guarantee: a NEW conversation greets, whatever the
            # nudge cascade did to the draft (observed live: a grounding
            # rewrite dropped the intro). Renewals greet when the client is
            # actually starting a new order flow.
            must_greet = intro_mode == "new" or (
                intro_mode == "renew" and (executor.order_activity
                                           or executor.results_this_turn))
            if must_greet and "kapa" not in rep.lower():
                rep = (f"¡Hola {client_name}! Soy Kapa, el asistente de "
                       f"{salesman_name}. ") + rep.lstrip()
            elif intro_mode == "ongoing":
                # Symmetric guarantee: mid-conversation turns never
                # re-introduce (observed live: identical intro replayed).
                low = _strip_accents_lower(rep[:160])
                if "soy kapa" in low:
                    idx = low.index("soy kapa")
                    cut = len(rep)
                    for stop in (".", "!", "\n"):
                        p = rep.find(stop, idx)
                        if p != -1:
                            cut = min(cut, p + 1)
                    stripped = rep[cut:].lstrip(" 👋😊✨🙌\n")
                    if stripped:
                        logger.info("[AGENT] Stripped a re-introduction on an ongoing turn.")
                        rep = stripped
        existing = {i.get("pedido") for i in executor.open_items}
        for h in executor.held_for_enrichment:
            key = h.strip().lower()
            if key and key not in existing:
                executor.open_items.append({"pedido": key, "estado": "enriquecer",
                                            "opciones": []})
        """Every exit runs the final grounding pass: deferred add_item calls
        (codes confirmed from a previous turn, unseen in this one) whose
        grounding search never ran — e.g. the confirmation landed on the LAST
        loop step after a nudge consumed a round — get one out-of-budget
        exact-code search here. A client-confirmed item must never be lost to
        step accounting; if the code genuinely isn't in the catalog, that is
        logged as an ERROR instead of vanishing silently."""
        if executor.pending_adds:
            codes = list(executor.pending_adds)
            logger.warning(f"[AGENT] Final grounding pass for deferred adds: {codes}")
            try:
                executor.register_results(await search(codes))
            except Exception as e:
                logger.error(f"[AGENT] Final grounding search failed: {e}")
            if executor.pending_adds:
                logger.error(f"[AGENT] CONFIRMED ITEMS DROPPED (codes not found in catalog): "
                             f"{dict(executor.pending_adds)}")
        r = _res(rep)
        if r.order_status == "CLOSED":
            logger.info(f"[AGENT] Order CLOSED with {len(r.cart)} items: "
                        f"{[(i['code'], i['qty']) for i in r.cart]}")
        return r

    reply = ""
    llm = _get_client()
    demoted_handoff = False
    demoted_optout = False
    demoted_promise = False
    demoted_style = False
    demoted_humility = False
    demoted_media_add = False
    demoted_intro = False
    demoted_generic_q = False
    demoted_mixed = False
    demoted_grounding = False
    budget = MAX_STEPS
    step = -1
    while step + 1 < budget:
        step += 1
        ctx_stats["used"] = max(ctx_stats["used"],
                                sum(_est_tokens(m.content or "") for m in messages))
        try:
            resp = await asyncio.to_thread(
                llm.chat, CIMA_MODEL, messages, tools=_TOOLS,
                options={"temperature": AGENT_TEMPERATURE, "top_p": 0.9, "repeat_penalty": 1.1,
                         "num_predict": 512, "stop": ["<NO_REPLY>"]},
            )
        except Exception as e:
            logger.error(f"[AGENT] LLM call failed at step {step}: {e}")
            return await _finish("")

        reply = _clean(resp.content)
        queries = executor.apply(resp.tool_calls or [])
        logger.info(f"[AGENT] step={step} tools={len(resp.tool_calls or [])} "
                    f"queries={queries} status={executor.status} cart={len(executor.cart)} "
                    f"handoff={executor.handoff} opt_out={executor.opt_out}")

        # Escalations: honor when the client's message corroborates them, when
        # the model insists (second call), or when no nudge round remains.
        can_nudge = step < budget - 1
        if executor.opt_out and can_nudge and not demoted_optout and \
                not _message_has_signals(current_message, _OPTOUT_SIGNAL_WORDS):
            executor.opt_out = False
            demoted_optout = True
            logger.info("[AGENT] opt_out demoted (no rejection signals in the client message).")
            messages.append(Message(role="user", content=_internal(
                "No consta que el cliente rechace hablar con un asistente. Responde tú mismo a su "
                "mensaje (o llama a opt_out_client otra vez SOLO si realmente no quiere hablar contigo).")))
            continue
        if executor.handoff and can_nudge and not demoted_handoff and \
                not _message_has_signals(current_message, _HANDOFF_SIGNAL_WORDS, extra_words=salesman_name):
            executor.handoff = False
            demoted_handoff = True
            logger.info("[AGENT] handoff demoted (no escalation signals in the client message).")
            messages.append(Message(role="user", content=_internal(
                f"No consta que el cliente pida hablar con {salesman_name} ni pregunte por precios, "
                "stock o incidencias. Responde tú mismo siguiendo tus reglas (o llama a "
                "handoff_to_human otra vez SOLO si de verdad no puedes ayudarle).")))
            continue
        if executor.handoff and not executor.opt_out and can_nudge and not demoted_mixed \
                and (queries or executor.order_activity or executor.blocked_media_adds
                     or executor.blocked_media_close):
            # MIXED INTENT (observed live): "añade dos de la caja... y dime el
            # precio" — the corroborated handoff must NOT strand the order.
            # The handoff flag is CLEARED: notification emails are gone (product
            # decision) and a mixed turn is an order turn; the price referral is
            # said in Kapa's own words instead of the canned message.
            executor.handoff = False
            demoted_mixed = True
            blocked = ", ".join(dict.fromkeys(executor.blocked_media_adds)) or "los productos mencionados"
            executor.blocked_media_adds.clear()
            logger.info("[AGENT] Mixed intent (order + escalation): keeping the order alive.")
            messages.append(Message(role="assistant", content=reply or "(handoff)"))
            messages.append(Message(role="user", content=_internal(
                f"El cliente mezcla PEDIDO y una consulta para {salesman_name} (precio/envío...). "
                f"No abandones el pedido: di breve que esa consulta se la pasas a {salesman_name} "
                f"y sigue con el pedido — propón lo que entendiste ({blocked}) y espera su 'sí' "
                f"antes de apuntar nada.")))
            continue
        if executor.handoff or executor.opt_out:
            return await _finish("")  # pure escalation: caller sends the canned message
        # A demoted mixed turn falls through as a normal order turn: the reply
        # still passes the grounding/humility/parroting guards. If the model
        # insists on handoff_to_human in the NEXT round, it is honored.

        if reply and not queries and can_nudge and not demoted_promise and _promises_search(reply):
            demoted_promise = True
            logger.info("[AGENT] Reply promised a search without calling the tool; nudging.")
            messages.append(Message(role="assistant", content=reply))
            messages.append(Message(role="user", content=_internal(
                "Has dicho que vas a buscar pero NO has llamado a search_products. Llámala AHORA "
                "con la consulta adecuada (combina la marca o referencia del cliente con el "
                "producto del que hablabais en el historial), o responde con lo que ya sabes "
                "sin prometer búsquedas.")))
            continue

        if reply and not queries and can_nudge and not demoted_style and _is_parroting(reply, recent_history):
            demoted_style = True
            logger.info("[AGENT] Reply parrots a previous message verbatim; asking for a rewrite.")
            messages.append(Message(role="assistant", content=reply))
            messages.append(Message(role="user", content=_internal(
                "Estás repitiendo literalmente frases de tus mensajes anteriores. Reescribe la "
                "respuesta con otras palabras: reacciona a lo que acaba de decir el cliente, "
                "mantén la lista numerada con los MISMOS códigos y termina con una pregunta distinta.")))
            continue

        if reply and not queries and can_nudge and not demoted_humility \
                and _is_media_message(current_message) \
                and _proposal_codes(reply) and not _invites_correction(reply):
            demoted_humility = True
            logger.info("[AGENT] Media-derived proposal lacks a correction invitation; nudging.")
            messages.append(Message(role="assistant", content=reply))
            messages.append(Message(role="user", content=_internal(
                "Tu propuesta viene de interpretar una imagen/audio: dilo con naturalidad (puedes "
                "haber leído mal) e invita al cliente a corregirte con más detalle, la marca o el "
                "código exacto. Reescribe variando las palabras, con los MISMOS códigos.")))
            continue

        if (executor.blocked_media_adds or executor.blocked_media_close or
                executor.blocked_empty_close) and can_nudge and not demoted_media_add:
            demoted_media_add = True
            parts: List[str] = []
            if executor.blocked_media_adds:
                codes = ", ".join(dict.fromkeys(executor.blocked_media_adds))
                parts.append(f"el mensaje es una lectura automática (audio/imagen) y NO contiene "
                             f"el código {codes}: no lo apuntes todavía; di qué has entendido, "
                             f"propón el producto y espera su confirmación")
            if executor.blocked_media_close:
                parts.append("tampoco cierres el pedido desde un audio/imagen: resume lo entendido "
                             "más el carrito actual y pide la confirmación final por escrito")
            if executor.blocked_empty_close:
                parts.append("el pedido está vacío: no se puede cerrar; pregunta qué productos quiere")
            executor.blocked_media_adds.clear()
            executor.blocked_media_close = False
            executor.blocked_empty_close = False
            messages.append(Message(role="assistant", content=reply or "(acción bloqueada)"))
            messages.append(Message(role="user", content=_internal("; ".join(parts).capitalize() + ".")))
            continue

        if (reply and not queries and can_nudge and not demoted_intro
                and intro_mode in ("new", "renew")
                and len((current_message or "").split()) >= 5
                and not executor.results_this_turn
                and not executor.held_for_enrichment and not demoted_generic_q
                and not _proposal_codes(reply) and not executor.order_activity):
            demoted_intro = True
            logger.info("[AGENT] Intro reply ignored the message content; nudging.")
            messages.append(Message(role="assistant", content=reply))
            messages.append(Message(role="user", content=_internal(
                "El cliente YA ha dicho lo que necesita en su mensaje: atiéndelo en esta misma "
                "respuesta (busca los productos que nombra o responde a lo que plantea), además "
                "del saludo. No le preguntes qué necesita.")))
            continue

        # PROPOSAL GROUNDING: every code in a numbered proposal must come from
        # THIS turn's search results (or the cart; or the last closed order when
        # the client asks to repeat it). Anything else — Memoria, old history,
        # prompt examples — presented as "found" is inventing availability.
        if reply and not queries:
            allowed = set(executor.results_this_turn) | {_sane_code(c) for c in executor.cart}
            if _message_has_signals(current_message, _REPEAT_SIGNAL_WORDS):
                allowed |= {_sane_code(i.get("code", "")) for i in (session.get("last_closed_cart") or [])}
            bad = {c for c in _proposal_codes(reply) if c and c not in allowed}
            if bad:
                if can_nudge and not demoted_grounding:
                    demoted_grounding = True
                    logger.info(f"[AGENT] Ungrounded proposal codes {sorted(bad)}; asking for a rewrite.")
                    messages.append(Message(role="assistant", content=reply))
                    messages.append(Message(role="user", content=_internal(
                        f"Has propuesto códigos que NO están en los resultados de esta búsqueda: "
                        f"{', '.join(sorted(bad))}. Reescribe la propuesta usando SOLO los resultados "
                        "actuales (busca de nuevo si lo necesitas). No ofrezcas nada que no haya salido; "
                        "los códigos de ejemplo de tus instrucciones NO existen.")))
                    continue
                logger.warning(f"[AGENT] Stripping ungrounded proposal codes {sorted(bad)} from the reply.")
                reply = _strip_ungrounded_lines(reply, bad)

        if queries and step >= budget - 1 and not executor.results_this_turn \
                and budget == MAX_STEPS:
            budget += 1     # grace: run the turn's first search + one phrasing call
            logger.info("[AGENT] Grace round granted for a last-step first search.")
        if queries and step < budget - 1:
            enrich_pending = {i.get("pedido") for i in executor.open_items
                              if i.get("estado") == "enriquecer"}
            first_talk = not session.get("guide_shown")
            runnable, held = [], []
            for q in dict.fromkeys(queries):
                if q.strip().lower() in enrich_pending:
                    runnable.append(q)      # client already engaged on this item
                    continue
                if _too_generic(q) or (first_talk and not _code_shaped(q)):
                    held.append(q)
                else:
                    runnable.append(q)
            if held:
                logger.info(f"[AGENT] Generic queries HELD (no Qdrant call): {held}")
                executor.held_for_enrichment.extend(
                    h for h in held if h not in executor.held_for_enrichment)
            if not runnable:
                if can_nudge and not demoted_generic_q:
                    demoted_generic_q = True
                    messages.append(Message(role="assistant", content=json.dumps(
                        {"tool_calls": [{"search_products": queries}]}, ensure_ascii=False)))
                    if first_talk:
                        nudge = (
                            f"PRIMER pedido: NO busques aún. Saluda si toca y explica en corto "
                            f"cómo trabajas: el código de producto es lo más fácil (lo verificas al "
                            f"momento); si no, buscas por nombre. Resume lo que entiendes que "
                            f"quiere (con cantidades: {', '.join(held)}) y dile que en cuanto te "
                            f"lo confirme o te pase códigos de producto, lo buscas. Termina "
                            f"preguntando cómo prefiere seguir, sin muletillas.")
                    else:
                        nudge = (
                            f"Consulta demasiado genérica ({', '.join(held)}): saldrían cientos "
                            f"de resultados. Si el mensaje o el historial dan más contexto "
                            f"(marca, uso, color), busca con ESO. Si no, pide el código de producto "
                            f"(lo más fácil) o nombre casi exacto/palabras clave y tú buscas "
                            f"(sugiere un ejemplo). Si dice que se llama así tal cual, lo buscarás.")
                    messages.append(Message(role="user", content=_internal(nudge)))
                    continue
                queries = []
                reply = reply or ""
                break
            queries = runnable
            results = await search(queries)
            executor.register_results(results)
            # Feed a TRIAGED digest, not a raw hit dump: resolved items in one
            # line each, ONE question at a time, the rest queued — how a real
            # salesman walks a client's list. The queue persists in the session.
            digest, executor.open_items = _triage_results(
                results, executor.open_items,
                is_media=_is_media_message(current_message))
            if executor.held_for_enrichment:
                digest["necesitan_detalle"] = list(executor.held_for_enrichment)
                digest["instruccion_detalle"] = (
                    "para estos pide, con humildad, nombre exacto o palabras clave "
                    "(sugiere tú un ejemplo de búsqueda más completa); el código es lo más "
                    "exacto; si de verdad se llama así tal cual, lo buscarás")
            # A runnable query that covers a pending-enrichment item means the
            # client already gave the detail: clear the stale item.
            ran_tokens = {t for q in queries for t in _query_tokens(q)}
            executor.open_items = [
                i for i in executor.open_items
                if i.get("estado") != "enriquecer"
                or not (set(_query_tokens(i.get("pedido", ""))) & ran_tokens)
            ]
            messages.append(Message(role="assistant", content=json.dumps(
                {"tool_calls": [{"search_products": queries}]}, ensure_ascii=False)))
            messages.append(Message(role="tool", content=_cap_middle(
                json.dumps({"resultados_busqueda": digest}, ensure_ascii=False), MAX_RESULTS_CHARS)))
            continue

        if not reply and resp.tool_calls and step < MAX_STEPS - 1:
            # Small models often emit only tool calls first; ask for the final text.
            messages.append(Message(role="assistant", content=json.dumps(
                {"tool_calls": [c.get("function", {}).get("name") for c in resp.tool_calls]})))
            messages.append(Message(role="user",
                                    content="Cambios aplicados. Escribe AHORA tu respuesta final al cliente (o <NO_REPLY>)."))
            continue

        break

    reply, leaked_note = _split_leaked_note(reply)
    if leaked_note and not executor.summary_touched:
        executor.summary = leaked_note[:300]
        logger.info(f"[AGENT] Leaked visible note rerouted to summary: '{leaked_note[:60]}'")
    return await _finish(reply)