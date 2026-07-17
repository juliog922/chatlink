# chatlink_bot/src/chatlink_bot/ai/llm.py
"""
Kapa, as a TWO-PASS orchestrator: one cheap deterministic triage helper,
then the tool-driven agent loop.

PASS 1 — INTENT & SPECIFICITY TRIAGE (temperature 0.0, format=json).
A tiny isolated inference before the agent loop parses the compound client
message ("Quiero los 5 delineadores, 4 cremas para la cara y 3 L'ACTION
mascarilla facial antiedad 8 grs") into structured intents:

    escalada    the client explicitly demands the human salesman, shows real
                anger, or raises an uncommercial dispute. When the message
                itself corroborates it, the orchestrator SHORT-CIRCUITS with
                strict silence (<NO_REPLY> semantics: reply="" + handoff=True
                + silent=True) so the salesman takes over without a canned
                bot line. Uncorroborated flags are demoted to a hint the
                agent sees (it can still call handoff_to_human).
    WORTHY      product mentions carrying a brand (L'Action, Nivea…), line
                attributes, exact measurements (8 grs, 150 ml) or a code —
                specific enough to hit Qdrant IMMEDIATELY, even on turn one.
    AMBIGUOUS   bare category talk ("crema de cara", "unos delineadores"):
                never searched blind; queued in open_items under
                estado="enriquecer" and Kapa amably asks for detail,
                emphasizing that the código de producto is the fastest,
                safest way to secure the item.

This REPLACES every hardcoded semantic word list. The `first_talk` code-only
gate (which swept high-quality descriptions into `held` and spat a robotic
canned speech) is gone. So are ALL keyword/vocabulary heuristics —
generic-noun sets, handoff/opt-out signal words, repeat-order signal words,
attribute-color lists, correction-phrase tuples, search-promise regexes.
Every semantic judgement is now made by one of THREE isolated temperature-0.0
helpers, keeping each individual prompt tiny (< 70% context budget):

    run_triage       (per turn)      message-level: escalation + evidence,
                                     rechazo_bot, derivacion, repetir_ultimo,
                                     per-item WORTHY/AMBIGUOUS/ATTRIBUTE.
    run_query_gate   (on demand)     query-level: SEARCHABLE / GENERIC /
                                     ATTRIBUTE for mid-loop queries Pass-1
                                     never saw. Fails PERMISSIVE (run it).
    run_reply_audit  (on demand)     draft-level: announced-but-not-executed
                                     searches; humility invitation on media
                                     proposals. Fails PERMISSIVE (skip nudge).

What remains in code is purely STRUCTURAL, never vocabulary: code shapes
(compact token with a digit), proposal-line parsing ('N) CODE — NAME'),
verbatim-repetition detection, pipeline media markers our own handlers
inject, evidence-quote containment, and token-overlap fuzzy matching.

PASS 2 — THE AGENT TURN (temperature 0.55).
The Pass-1 JSON is injected into Kapa's user context block as a "TRIAJE
PREVIO" section, and it gates retrieval in code: WORTHY / code-shaped /
pending-enrichment queries run against Qdrant; AMBIGUOUS ones are held and
queued; ATTRIBUTE ones are grounded with the structural topic from the bot's
own last proposal. Everything else is the proven single agent loop:

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
import time
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

# How many of the bot's own recent turns to scan for a codes mention before
# gently re-surfacing the "codes are the fastest/safest" tip. Larger = rarer.
CODES_TIP_WINDOW = int(os.getenv("AGENT_CODES_TIP_WINDOW", "4"))

# Every non-empty reply carries a short signature so the client can never
# mistake Kapa for the human salesman. Deterministic (appended in code, not
# left to the 2B model). Configurable; keep it short.
BOT_SIGNATURE = os.getenv("AGENT_BOT_SIGNATURE", "— Kapa 🤖")

# ---- Pass-1 triage helper (isolated, single-task, deterministic) -----------
TRIAGE_TEMPERATURE = 0.0                                            # never configurable: determinism is the contract
TRIAGE_TIMEOUT_S = float(os.getenv("AGENT_TRIAGE_TIMEOUT_S", "90"))
TRIAGE_MAX_TOKENS = int(os.getenv("AGENT_TRIAGE_MAX_TOKENS", "1400"))  # long lists need room
TRIAGE_MSG_MAX_CHARS = int(os.getenv("AGENT_TRIAGE_MSG_MAX_CHARS", "1600"))
TRIAGE_HISTORY_MAX_CHARS = int(os.getenv("AGENT_TRIAGE_HISTORY_MAX_CHARS", "600"))

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
    # Triage-driven escalation (Pass 1): handoff WITHOUT the canned message —
    # strict silence so the human salesman takes over. handlers.py checks it.
    silent: bool = False
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
den el nombre — y sigue con el resto. En el historial hay TRES voces: \
"Cliente:" es el cliente, "Asistente:" eres TÚ, y "Comercial:" es \
{salesman_name} en persona. Lee la conversación ENTERA: si el Comercial ya \
respondió algo, dalo por dicho — no lo repitas ni lo contradigas; si él dejó \
un pedido a medias y el cliente sigue, continúa desde donde él lo dejó. No \
te confundas de quién dijo qué ni repitas tus propias frases. Si ya te \
presentaste en el historial, no saludes ni te presentes.

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

SI PIDEN VER EL CATÁLOGO o "¿qué productos tienes?" / "¿qué hay?": NO es una \
derivación ni un motivo para pasar al comercial. Explícalo natural: no puedes \
mostrar el catálogo entero (son miles de productos), PERO si te dan el nombre \
completo o el código de producto lo localizas al instante. Pídele que te diga \
qué busca y ofrécete a buscarlo. Nunca respondas a esto con un "te paso con \
{salesman_name}".

ANTES DE DERIVAR, di lo que SÍ puedes: solo cuando el cliente pida algo que \
de verdad no te toca (precios/descuentos, stock, facturas/pagos, \
incidencias/devoluciones, envío o estado de un pedido enviado, o consejo de \
producto) NO uses el catálogo para inventar: dile con tus palabras que ESO se \
lo confirma {salesman_name}, y recuérdale que tú sí puedes ayudarle a montar \
el pedido y a encontrar códigos. Si MEZCLA pedido y precio/duda: atiende el \
pedido y menciona de pasada lo del comercial. handoff_to_human SOLO cuando \
no haya nada del pedido que puedas atender, o cuando el cliente pida \
explícitamente hablar con una persona (entonces te callas y entra él).
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
{triage_block}
### MENSAJE(S) NUEVO(S) DEL CLIENTE
{current_message}

Actualiza el estado con las herramientas que hagan falta y escribe tu respuesta \
final para el cliente (o <NO_REPLY>)."""


# ------------------------------------------------------------------- executor
_re_query_tokens = re.compile(r"[^\w]+", re.UNICODE)

# Escalation tools (handoff_to_human / opt_out_client) are honored immediately
# ONLY when the Pass-1 triage's independent temperature-0.0 read of the client
# message corroborates them (`derivacion` / `rechazo_bot`) — observed live:
# "¿cómo hago un pedido?" triggered handoff_to_human because 'pedido' smells
# like salesman territory to a 2B model, bouncing a trivial process question
# to a human. Uncorroborated calls are demoted once with a nudge to answer
# directly; if the model INSISTS on a second call, it is honored (covers real
# cases no classifier can foresee). No keyword lists anywhere: two model
# opinions at different temperatures must agree, or the model must insist.


def _strip_accents_lower(s: str) -> str:
    import unicodedata
    s = (s or "").lower()
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


# A reply that ANNOUNCES a search without calling the tool leaves the client
# waiting forever — observed live: "Busca un momento, reviso qué tengo" with
# zero tool calls. Judged by the reply-audit helper (run_reply_audit, a
# temperature-0.0 pass over the draft reply), not by a phrase regex.


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
# the last closed order when the Pass-1 triage says the client asked to
# REPEAT their usual order — `repetir_ultimo`, no keyword matching). One
# rewrite nudge; then offending lines are stripped and the list renumbered.
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
# clips. WHICH queries are attribute-only is judged by the helpers (Pass-1
# `clase: ATTRIBUTE`, or the query-gate helper for mid-loop queries — no
# color/size word list). The current product family is still derived
# STRUCTURALLY from the bot's own last numbered proposal in the history
# (tokens repeated across its items, e.g. DELINEADOR/KATAI), so attribute
# queries are grounded in code: 'negro' -> 'delineador katai negro'. No topic
# available -> the item is queued and the model asks which product it refers to.
_re_proposal_item = re.compile(r"\d+\)\s*[A-Za-z0-9\-./]+\s*—\s*([^0-9)][^)]{2,80}?)(?=\s+\d+\)|$)")


def _topic_from_history(recent_history: str) -> List[str]:
    """Product-family tokens from the newest bot proposal in the history:
    purely structural — tokens (len >= 4) REPEATED across >= 2 item names of
    the same numbered list. Repetition across variants of one family is what
    identifies the family ('DELINEADOR KATAI NEGRO' / 'DELINEADOR KATAI AZUL'
    -> ['delineador', 'katai']); no vocabulary list involved. No proposal =
    no established product context."""
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
                if len(t) >= 4
            })
        # Tokens shared by ALL items first (the family), then by most.
        common = [t for t, c in counts.most_common(6) if c >= 2][:2]
        if common:
            return common
    return []


def _should_remind_codes(recent_history: str) -> bool:
    """From-time-to-time codes tip, statelessly: True when the bot has NOT
    mentioned 'código' in its last CODES_TIP_WINDOW turns. Right after it does,
    the mention sits in the recent window and suppresses the tip until it ages
    out — a natural cadence with no counter column to persist. Reads only the
    bot's own lines for its own domain term (not client-intent classification)."""
    bot_lines = [l for l in (recent_history or "").splitlines() if l.startswith("Asistente:")]
    if not bot_lines:
        return False   # first contact: the one-time how-it-works guide covers codes
    recent = " ".join(bot_lines[-CODES_TIP_WINDOW:])
    return "codigo" not in _strip_accents_lower(recent)


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
# brand, or the exact code). Whether the draft actually invites correction is
# judged SEMANTICALLY by the reply-audit helper (run_reply_audit, temperature
# 0.0) — no phrase list — so the wording stays the model's own and no fixed
# formula can calcify into a template. One nudge; never fires on plain-text
# turns. _MEDIA_MARKERS are NOT client language: they are literal protocol
# markers OUR OWN pipeline (handlers._MEDIA_PREFIX) injects — structural.
_MEDIA_MARKERS = ("[Texto en Imagen]", "[Audio transcrito]", "[Documento")


def _is_media_message(message: str) -> bool:
    return any(m in (message or "") for m in _MEDIA_MARKERS)


# ------------------------------------------------ Pass 1: triage helper
# An isolated, single-task inference at temperature 0.0 that runs BEFORE the
# agent loop. It answers exactly three questions about the raw client
# message: (a) is this an escalation the bot must stay out of, (b) which
# product mentions are specific enough to search immediately (WORTHY), and
# (c) which are generic category talk that needs conversational narrowing
# (AMBIGUOUS). Multi-round helpers like this one are the house pattern:
# each prompt stays tiny (well under the 70% context budget) and a 2B model
# does one job well instead of many jobs badly.

TRIAGE_ESC = "ESC_HANDOFF"
TRIAGE_WORTHY = "WORTHY"
TRIAGE_AMBIGUOUS = "AMBIGUOUS"
TRIAGE_ATTRIBUTE = "ATTRIBUTE"


@dataclass(frozen=True)
class TriageItem:
    mention: str        # verbatim-ish client mention ("3 L'ACTION mascarilla…")
    query: str          # clean search terms ("l'action mascarilla facial antiedad 8 grs")
    qty: int
    cls: str            # WORTHY | AMBIGUOUS | ATTRIBUTE


@dataclass
class TriageResult:
    ok: bool = False                 # False -> helper failed; gates fall back
    escalation: bool = False
    evidence: str = ""               # verbatim quote backing the escalation
    opt_out: bool = False            # client rejects talking to a bot/assistant
    refer_salesman: bool = False     # commercial question bot can't answer -> spoken referral
    greeting: bool = False           # greeting / order-opener -> always respond, never silence
    small_talk: bool = False         # purely social/off-topic -> silence
    repeat_order: bool = False       # asks to repeat the usual / last order
    items: List[TriageItem] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def worthy(self) -> List[TriageItem]:
        return [i for i in self.items if i.cls == TRIAGE_WORTHY]

    @property
    def ambiguous(self) -> List[TriageItem]:
        return [i for i in self.items if i.cls == TRIAGE_AMBIGUOUS]

    @property
    def attributes(self) -> List[TriageItem]:
        return [i for i in self.items if i.cls == TRIAGE_ATTRIBUTE]


_TRIAGE_SYSTEM = """\
Eres un clasificador determinista de mensajes de clientes de una tienda de \
cosmética. NO conversas: devuelves SOLO un objeto JSON válido, sin texto \
fuera del JSON, con esta forma exacta:
{"escalada": false, "evidencia": "", "rechazo_bot": false, "derivacion": \
false, "saludo": false, "charla_no_comercial": false, "repetir_ultimo": \
false, "articulos": [{"mencion": "…", "consulta": "…", "cantidad": 1, \
"clase": "WORTHY"}]}

REGLAS DE LOS INDICADORES (true SOLO si el mensaje NUEVO lo dice claramente):
- "escalada": el cliente PIDE hablar con una persona / el comercial / un \
humano, o muestra enfado real (insultos, amenaza de baja, hartazgo \
explícito). En ese caso "evidencia" = cita LITERAL del fragmento del mensaje. \
Preguntar precios o por un pedido NO es escalada; saludar o pedir productos \
tampoco.
- "rechazo_bot": expresa que NO quiere hablar con un asistente/bot/máquina \
("no quiero hablar con un robot", "deja de escribirme").
- "derivacion": pregunta o pide algo COMERCIAL que solo resuelve el comercial \
humano y que el asistente NO debe responder: precios, descuentos, stock, \
facturas, pagos, incidencias, reclamaciones, devoluciones, estado o envío de \
un pedido, o consejo sobre productos (para qué sirve, cuál es mejor). PEDIR \
productos ("quiero 4 champús", "ponme 2 cremas") NO es derivacion: eso es un \
pedido normal, va en "articulos". Preguntar "¿qué productos tienes?" o pedir \
ver el catálogo TAMPOCO es derivacion: el asistente lo resuelve él mismo \
(busca por nombre/código), así que marca "saludo" en su lugar. El asistente \
NO se calla en derivacion: dirá que eso se lo confirma el comercial.
- "saludo": el mensaje es un saludo o una apertura de conversación ("hola", \
"buenas", "hey", "¿estás?", "buenos días") o dice que quiere hacer/empezar un \
pedido SIN concretar productos todavía ("quiero hacer un pedido", "necesito \
encargar unas cosas"). Esto SÍ se responde: es intención comercial o el \
principio de una. NUNCA lo marques también como charla_no_comercial.
- "charla_no_comercial": SOLO conversación personal o ajena al negocio, SIN \
saludo, SIN pedido y SIN pregunta comercial: interesarse por ti o por el \
comercial ("¿qué tal estás?", "¿cómo está tu madre?", "¿qué hiciste ayer?"), \
o temas que no son de la tienda ("¿cuál es la capital de Francia?"). Un \
simple "hola" NO es charla (es "saludo"); querer pedir algo NO es charla.
- "repetir_ultimo": pide repetir su pedido anterior o habitual ("lo de \
siempre", "ponme lo mismo que la última vez").

REGLAS DE "articulos" (una entrada por producto mencionado; [] si no hay):
- "mencion": el fragmento del cliente, con su cantidad si la dice.
- "consulta": términos de búsqueda cortos: producto + marca/atributos/medida. \
Sin cantidades, sin verbos, sin frases enteras.
- "cantidad": el número pedido (1 si no lo dice).
- "clase" = "WORTHY" si lleva marca (L'Action, Nivea, Katai…), medida exacta \
(8 grs, 150 ml, nº 3), atributos concretos de línea (antiedad, waterproof, \
tono nude…) o un código alfanumérico (KG001399, 14-1127) — con eso ya se \
puede buscar en el catálogo.
- "clase" = "AMBIGUOUS" si es categoría genérica sin marca ni medida ni \
atributo distintivo ("una crema de cara", "unos delineadores", "algo para \
el pelo") — buscar eso a ciegas devolvería ruido.
- "clase" = "ATTRIBUTE" si la mención es SOLO un atributo o variante — un \
color, un tamaño, un acabado, un número de la lista — del producto del que \
se venía hablando en el historial ("el negro", "la grande", "el mate").

EJEMPLOS
Mensaje: "Hola"
Respuesta: {"escalada": false, "evidencia": "", "rechazo_bot": false, \
"derivacion": false, "saludo": true, "charla_no_comercial": false, \
"repetir_ultimo": false, "articulos": []}
Mensaje: "Quiero hacer un pedido"
Respuesta: {"escalada": false, "evidencia": "", "rechazo_bot": false, \
"derivacion": false, "saludo": true, "charla_no_comercial": false, \
"repetir_ultimo": false, "articulos": []}
Mensaje: "¿Cómo está tu madre?"
Respuesta: {"escalada": false, "evidencia": "", "rechazo_bot": false, \
"derivacion": false, "saludo": false, "charla_no_comercial": true, \
"repetir_ultimo": false, "articulos": []}
Mensaje: "Quiero 4 champús y 2 cremas"
Respuesta: {"escalada": false, "evidencia": "", "rechazo_bot": false, \
"derivacion": false, "saludo": false, "charla_no_comercial": false, \
"repetir_ultimo": false, "articulos": [\
{"mencion": "4 champús", "consulta": "champú", "cantidad": 4, "clase": \
"AMBIGUOUS"}, {"mencion": "2 cremas", "consulta": "crema", "cantidad": 2, \
"clase": "AMBIGUOUS"}]}"""

_TRIAGE_USER = """\
Historial reciente (solo contexto, NO lo clasifiques):
{history}

MENSAJE NUEVO DEL CLIENTE (clasifica SOLO esto):
{message}

Devuelve el JSON."""


def _parse_triage_json(raw: str) -> Optional[TriageResult]:
    """Strict-but-forgiving parse. On a long list the model may hit the token
    cap and cut the JSON off mid-array; rather than lose the whole turn, we
    salvage every COMPLETE object already emitted. None only when nothing
    usable can be recovered."""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("{")
    if start == -1:
        return None
    data: Optional[dict] = None
    end = text.rfind("}")
    if end > start:
        try:
            data = json.loads(text[start:end + 1])
        except Exception:
            data = None
    if not isinstance(data, dict):
        # Truncated output: reconstruct from the flags before "articulos" and
        # every complete {...} object inside the (unterminated) array.
        data = {}
        head = text[start:]
        for flag in ("escalada", "rechazo_bot", "derivacion", "saludo",
                     "charla_no_comercial", "repetir_ultimo"):
            m = re.search(rf'"{flag}"\s*:\s*(true|false)', head)
            if m:
                data[flag] = m.group(1) == "true"
        mev = re.search(r'"evidencia"\s*:\s*"([^"]*)"', head)
        if mev:
            data["evidencia"] = mev.group(1)
        arr = head.find('"articulos"')
        objs: List[dict] = []
        if arr != -1:
            depth, obj_start = 0, -1
            for i in range(arr, len(head)):
                c = head[i]
                if c == "{":
                    if depth == 0:
                        obj_start = i
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0 and obj_start != -1:
                        try:
                            objs.append(json.loads(head[obj_start:i + 1]))
                        except Exception:
                            pass
                        obj_start = -1
        data["articulos"] = objs
        if not objs and not any(k in data for k in
                                ("escalada", "rechazo_bot", "derivacion", "saludo",
                                 "charla_no_comercial", "repetir_ultimo")):
            return None
        logger.info(f"[TRIAGE] Recovered {len(objs)} items from truncated output.")
    items: List[TriageItem] = []
    for it in (data.get("articulos") or []):
        if not isinstance(it, dict):
            continue
        query = str(it.get("consulta") or it.get("mencion") or "").strip()
        if not query:
            continue
        cls = str(it.get("clase") or "").strip().upper()
        if cls not in (TRIAGE_WORTHY, TRIAGE_AMBIGUOUS, TRIAGE_ATTRIBUTE):
            # Unknown label from the 2B model: judge by shape — codes are
            # always worthy; otherwise be conservative and ask.
            cls = TRIAGE_WORTHY if _code_shaped(query) else TRIAGE_AMBIGUOUS
        items.append(TriageItem(
            mention=str(it.get("mencion") or query).strip()[:120],
            query=query[:120],
            qty=_sane_qty(it.get("cantidad")),
            cls=cls,
        ))
    return TriageResult(
        ok=True,
        escalation=bool(data.get("escalada")),
        evidence=str(data.get("evidencia") or "").strip()[:200],
        opt_out=bool(data.get("rechazo_bot")),
        refer_salesman=bool(data.get("derivacion")),
        greeting=bool(data.get("saludo")),
        small_talk=bool(data.get("charla_no_comercial")),
        repeat_order=bool(data.get("repetir_ultimo")),
        items=items[:24],
    )


async def run_triage(current_message: str, recent_history: str) -> TriageResult:
    """Pass 1. Never raises: any failure returns TriageResult(ok=False) and
    the agent-loop gates fall back to the conservative heuristics."""
    msg = _cap_middle((current_message or "").strip(), TRIAGE_MSG_MAX_CHARS)
    if not msg:
        return TriageResult(ok=True)   # nothing to classify; not a failure
    hist = _tail_lines_to_fit(recent_history or "", TRIAGE_HISTORY_MAX_CHARS) or "(sin historial)"
    messages = [
        Message(role="system", content=_TRIAGE_SYSTEM),
        Message(role="user", content=_TRIAGE_USER.format(history=hist, message=msg)),
    ]
    t0 = time.perf_counter()
    try:
        resp = await asyncio.to_thread(
            _get_client().chat, CIMA_MODEL, messages,
            fmt="json",
            options={"temperature": TRIAGE_TEMPERATURE, "top_p": 1.0,
                     "num_predict": TRIAGE_MAX_TOKENS},
            timeout=TRIAGE_TIMEOUT_S,
        )
    except Exception as e:
        logger.warning(f"[TRIAGE] Helper call failed ({e}); falling back to heuristics.")
        return TriageResult(ok=False, elapsed_ms=(time.perf_counter() - t0) * 1000)
    parsed = _parse_triage_json(resp.content)
    elapsed = (time.perf_counter() - t0) * 1000
    if parsed is None:
        logger.warning(f"[TRIAGE] Unparseable helper output ({resp.content[:120]!r}); "
                       "falling back to heuristics.")
        return TriageResult(ok=False, elapsed_ms=elapsed)
    parsed.elapsed_ms = elapsed
    logger.info(f"[TRIAGE] ok in {elapsed:.0f}ms: escalada={parsed.escalation} "
                f"rechazo_bot={parsed.opt_out} derivacion={parsed.refer_salesman} "
                f"saludo={parsed.greeting} charla={parsed.small_talk} "
                f"repetir={parsed.repeat_order} "
                f"worthy={[i.query for i in parsed.worthy]} "
                f"ambiguous={[i.query for i in parsed.ambiguous]} "
                f"attribute={[i.query for i in parsed.attributes]}")
    return parsed


def _classify_query(query: str, triage: TriageResult) -> Optional[str]:
    """Match an agent-emitted search query against the Pass-1 items by token
    overlap; return that item's class, or None when the triage didn't cover
    it (agent-derived from history, or helper unavailable)."""
    if not triage.ok or not triage.items:
        return None
    q_tokens = set(_query_tokens(query))
    if not q_tokens:
        return None
    best_cls, best_score = None, 0.0
    for item in triage.items:
        i_tokens = set(_query_tokens(item.query)) | set(_query_tokens(item.mention))
        if not i_tokens:
            continue
        inter = q_tokens & i_tokens
        if not inter:
            continue
        score = len(inter) / min(len(q_tokens), len(i_tokens))
        if score > best_score:
            best_score, best_cls = score, item.cls
    return best_cls if best_score >= 0.5 else None


def _triage_qty(query: str, triage: TriageResult) -> int:
    """Quantity the client attached to the triage item this query matches."""
    q_tokens = set(_query_tokens(query))
    for item in triage.items:
        i_tokens = set(_query_tokens(item.query)) | set(_query_tokens(item.mention))
        if q_tokens and i_tokens and len(q_tokens & i_tokens) / min(len(q_tokens), len(i_tokens)) >= 0.5:
            return item.qty
    return 1


def _same_item(a: str, b: str) -> bool:
    """Two query strings that talk about the same product (token overlap)."""
    ta, tb = set(_query_tokens(a)), set(_query_tokens(b))
    if not ta or not tb:
        return a.strip().lower() == b.strip().lower()
    return len(ta & tb) / min(len(ta), len(tb)) >= 0.5


def _triage_block(triage: TriageResult, escalation_hint: bool) -> str:
    """The Pass-1 analysis injected into Kapa's user context block."""
    if not triage.ok or (not triage.items and not escalation_hint
                         and not triage.refer_salesman and not triage.repeat_order):
        return ""
    lines: List[str] = ["### TRIAJE PREVIO DEL MENSAJE (análisis automático; el cliente NO lo ve)"]
    if triage.worthy:
        lines.append("- Concretos, BUSCA YA con search_products (una consulta por artículo): " +
                     "; ".join(f'"{i.query}" (x{i.qty})' for i in triage.worthy))
    if triage.ambiguous:
        lines.append("- Genéricos, NO los busques: " +
                     "; ".join(f'"{i.query}" (x{i.qty})' for i in triage.ambiguous) +
                     ". Resume lo que entendiste (con cantidades) y pide amable un poco más "
                     "de detalle (marca, línea o medida); recuerda con humildad que el "
                     "código de producto es lo más rápido y seguro para apuntarlos sin error.")
    if triage.attributes:
        lines.append("- Variantes del producto en curso (búscalas COMBINADAS con el producto "
                     "del que hablabais en el historial): " +
                     "; ".join(f'"{i.query}" (x{i.qty})' for i in triage.attributes))
    if triage.repeat_order:
        lines.append("- El cliente pide REPETIR su pedido habitual: usa el 'Último pedido "
                     "ENVIADO' del contexto como base (add_item con esos códigos).")
    if triage.refer_salesman:
        lines.append("- El cliente pregunta algo COMERCIAL que tú NO puedes resolver (precio, "
                     "stock, factura, incidencia, estado/envío de un pedido, descuento o consejo "
                     "de producto). NO te quedes en silencio y NO lo inventes: dile con amabilidad "
                     "y en una frase que ESO no se lo puedes confirmar tú y que se lo dirá el "
                     "comercial; si además hay pedido, atiéndelo con normalidad. NO llames a "
                     "handoff_to_human solo por esto.")
    if escalation_hint:
        ev = f' ("{triage.evidence}")' if triage.evidence else ""
        lines.append(f"- Posible petición de hablar con una persona{ev}: SOLO si el cliente de "
                     "verdad pide un humano o está enfadado, usa handoff_to_human (te callarás y "
                     "entra el comercial); si no, atiéndele tú.")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------- helper: mid-loop query gate (0.0)
# The agent invents queries MID-LOOP (from history, from tool results) that
# Pass-1 never saw. Their searchability is judged by this second isolated
# helper — again a model at temperature 0.0, not a noun list. Failure mode is
# PERMISSIVE: an unjudged query runs, and the result triage (_query_verdict /
# uncovered-token honesty) absorbs any noise it returns.

_QUERY_GATE_SYSTEM = """\
Eres un clasificador determinista de consultas de búsqueda para un catálogo \
de miles de productos de cosmética. NO conversas: devuelves SOLO un objeto \
JSON, con esta forma exacta:
{"veredictos": [{"consulta": "…", "clase": "SEARCHABLE"}]}
Repite cada consulta EXACTAMENTE como te llega, una entrada por consulta.

CLASES:
- "SEARCHABLE": identifica un producto concreto — nombre distintivo, marca, \
línea, medida o código ("crema kerapro", "delineador katai", "14-1127").
- "GENERIC": solo una categoría o palabras vacías que devolverían cientos de \
resultados ("crema", "productos", "esmaltes", "algo barato", "el catálogo").
- "ATTRIBUTE": solo un atributo o variante — color, tamaño, acabado, número \
— que depende del producto del que se venía hablando ("negro", "el mate", \
"la grande", "la 2")."""


async def run_query_gate(queries: List[str]) -> Dict[str, str]:
    """Classify ad-hoc agent queries not covered by Pass-1.
    -> {query.lower(): SEARCHABLE|GENERIC|ATTRIBUTE}. {} on any failure —
    the gate then stays permissive (run the search; result triage copes)."""
    qs = [q for q in dict.fromkeys((x or "").strip() for x in queries) if q]
    if not qs:
        return {}
    payload = json.dumps({"consultas": qs}, ensure_ascii=False)
    try:
        resp = await asyncio.to_thread(
            _get_client().chat, CIMA_MODEL,
            [Message(role="system", content=_QUERY_GATE_SYSTEM),
             Message(role="user", content=payload)],
            fmt="json",
            options={"temperature": TRIAGE_TEMPERATURE, "top_p": 1.0,
                     "num_predict": TRIAGE_MAX_TOKENS},
            timeout=TRIAGE_TIMEOUT_S,
        )
        text = (resp.content or "").strip()
        start, end = text.find("{"), text.rfind("}")
        data = json.loads(text[start:end + 1]) if start != -1 and end > start else {}
    except Exception as e:
        logger.warning(f"[QUERY_GATE] Helper failed ({e}); gate stays permissive.")
        return {}
    out: Dict[str, str] = {}
    for v in (data.get("veredictos") or []):
        if not isinstance(v, dict):
            continue
        q = str(v.get("consulta") or "").strip().lower()
        c = str(v.get("clase") or "").strip().upper()
        if q and c in ("SEARCHABLE", "GENERIC", "ATTRIBUTE"):
            out[q] = c
    logger.info(f"[QUERY_GATE] {out}")
    return out


# ----------------------------------------- helper: reply audit (0.0)
# Two live-observed failure modes of the DRAFT reply, judged semantically
# instead of by phrase regexes / keyword tuples:
#   promete_busqueda   the text announces "voy a mirar / un momento" without
#                      having called search_products -> the client waits forever.
#   invita_correccion  media-derived proposals must own their fallibility and
#                      invite correction (detail, brand, or the product code).
# Failure mode is PERMISSIVE: if the audit is unavailable the guards are
# skipped — a missed style nudge degrades polish, never state or safety.

_AUDIT_SYSTEM = """\
Eres un auditor determinista de borradores de respuesta de un asistente de \
pedidos por WhatsApp. NO conversas: devuelves SOLO un objeto JSON:
{"promete_busqueda": false, "invita_correccion": false}

- "promete_busqueda": true si el texto ANUNCIA que va a buscar, mirar, \
revisar o comprobar algo, o pide al cliente esperar un momento, en lugar de \
dar ya el resultado.
- "invita_correccion": true si el texto reconoce de algún modo que puede \
haber leído o entendido mal (una imagen, un audio, una búsqueda) e invita \
al cliente a corregir o confirmar con más detalle, la marca o el código de \
producto."""


async def run_reply_audit(reply: str) -> Optional[Dict[str, bool]]:
    """-> {"promete_busqueda": bool, "invita_correccion": bool} | None."""
    if not (reply or "").strip():
        return None
    try:
        resp = await asyncio.to_thread(
            _get_client().chat, CIMA_MODEL,
            [Message(role="system", content=_AUDIT_SYSTEM),
             Message(role="user", content=f"BORRADOR:\n{reply[:1200]}\n\nDevuelve el JSON.")],
            fmt="json",
            options={"temperature": TRIAGE_TEMPERATURE, "top_p": 1.0, "num_predict": 80},
            timeout=TRIAGE_TIMEOUT_S,
        )
        text = (resp.content or "").strip()
        start, end = text.find("{"), text.rfind("}")
        data = json.loads(text[start:end + 1]) if start != -1 and end > start else None
    except Exception as e:
        logger.warning(f"[REPLY_AUDIT] Helper failed ({e}); guards skipped this turn.")
        return None
    if not isinstance(data, dict):
        return None
    return {"promete_busqueda": bool(data.get("promete_busqueda")),
            "invita_correccion": bool(data.get("invita_correccion"))}


# ------------------------------------------- helper: open-items reconcile (0.0)
# The thread breaks when nothing reconciles the RUNNING order against the new
# client message: items the client abandons ("olvida los delineadores") linger
# in the queue and get re-asked every turn, and a "sí / apunta 4" confirmation
# of an already-located item isn't recognised as a commit. Observed live: the
# bot kept re-surfacing forgotten items and asked "¿te apunto?" for an item it
# had just said was apuntado. This isolated 0.0 pass reads the running queue +
# the new message and returns, per queued item, what the CLIENT decided —
# drop, confirm (with quantity), or leave pending. No keyword matching for
# "olvida/quita/sí": the model reads intent; code only applies the verdict.

_RECONCILE_SYSTEM = """\
Eres un ayudante determinista que actualiza el estado de un pedido según el \
ÚLTIMO mensaje del cliente. NO conversas: devuelves SOLO un objeto JSON:
{"descartar": ["pedido…"], "confirmar": [{"pedido": "…", "codigo": "…", \
"cantidad": 1}]}

Te doy la lista de artículos EN CURSO (cada uno con su "pedido", su "estado" \
y, si ya está localizado, su "codigo") y el mensaje nuevo del cliente.

- "descartar": los "pedido" que el cliente ya NO quiere — dice que los olvides, \
los quita, los cancela, o cambia de idea sobre ellos. Copia el texto del \
"pedido" tal cual.
- "confirmar": los artículos YA localizados (estado "propuesto" o con \
"codigo") que el cliente ACABA de aprobar en este mensaje ("sí", "apunta esos", \
"ponme 4 de ese"). Incluye su "codigo" y la "cantidad" que diga el cliente \
(si no dice número, usa 1). NO confirmes nada que el cliente no haya aprobado.
- Si el cliente no descarta ni confirma nada, devuelve listas vacías.
- No inventes pedidos ni códigos que no estén en la lista que te doy."""


async def run_open_items_reconcile(
    current_message: str, open_items: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """-> {"descartar": [pedido…], "confirmar": [{pedido, codigo, cantidad}]}
    or None on any failure (callers then leave the queue untouched)."""
    queue_view = [
        {"pedido": i.get("pedido", ""), "estado": i.get("estado", ""),
         "codigo": (i.get("opciones") or [{}])[0].get("codigo", "")}
        for i in (open_items or []) if i.get("pedido")
    ]
    if not queue_view:
        return None
    payload = json.dumps({"articulos_en_curso": queue_view,
                          "mensaje_cliente": (current_message or "")[:TRIAGE_MSG_MAX_CHARS]},
                         ensure_ascii=False)
    try:
        resp = await asyncio.to_thread(
            _get_client().chat, CIMA_MODEL,
            [Message(role="system", content=_RECONCILE_SYSTEM),
             Message(role="user", content=payload)],
            fmt="json",
            options={"temperature": TRIAGE_TEMPERATURE, "top_p": 1.0, "num_predict": TRIAGE_MAX_TOKENS},
            timeout=TRIAGE_TIMEOUT_S,
        )
        text = (resp.content or "").strip()
        start, end = text.find("{"), text.rfind("}")
        data = json.loads(text[start:end + 1]) if start != -1 and end > start else None
    except Exception as e:
        logger.warning(f"[RECONCILE] Helper failed ({e}); queue left untouched.")
        return None
    if not isinstance(data, dict):
        return None
    descartar = [str(p).strip() for p in (data.get("descartar") or []) if str(p).strip()]
    confirmar = []
    for c in (data.get("confirmar") or []):
        if isinstance(c, dict) and (c.get("codigo") or c.get("pedido")):
            confirmar.append({"pedido": str(c.get("pedido") or "").strip(),
                              "codigo": _sane_code(c.get("codigo")),
                              "cantidad": _sane_qty(c.get("cantidad"))})
    logger.info(f"[RECONCILE] descartar={descartar} confirmar={confirmar}")
    return {"descartar": descartar, "confirmar": confirmar}
#   _query_tokens   tokenizer used for FUZZY MATCHING (query <-> triage item,
#                   query <-> pending-enrichment item, topic extraction). The
#                   tiny stopword set only stabilises overlap ratios; it never
#                   decides whether anything is searched, held, or escalated —
#                   every semantic verdict comes from the 0.0-temperature
#                   helpers (run_triage / run_query_gate / run_reply_audit).
#   _code_shaped    a STRUCTURAL shape test (compact token containing a
#                   digit), not vocabulary: a product code is a format, and
#                   codes are always searchable — the exact-code path in RAG
#                   is the judge of whether they exist.
# Recovery ladder for held (needs-detail) items, unchanged and list-free:
#   1. the model self-enriches from message/history context (one nudge),
#   2. the client is asked for name/keywords/code (item queued as
#      estado="enriquecer" so it is never lost),
#   3. the client insists the product is really called just that ->
#      the pending-enrichment bypass lets that exact query through.
_QUERY_STOP = {"de", "la", "el", "los", "las", "un", "una", "unos", "unas",
               "para", "por", "con", "y", "o", "del", "al", "en", "mi", "tu"}


def _query_tokens(q: str) -> List[str]:
    norm = _strip_accents_lower(q)
    return [t for t in re.split(r"[^a-z0-9ñ]+", norm)
            if len(t) >= 3 and t not in _QUERY_STOP and not t.isdigit()]


def _code_shaped(q: str) -> bool:
    q = (q or "").strip()
    return len(q) >= 4 and " " not in q and any(ch.isdigit() for ch in q)


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
                    is_media: bool = False,
                    held_count: int = 0) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
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
    # BATCH-QUALITY GATE (partial match is the LAST resort). When a whole
    # list comes in — especially a photo of a handwritten one — and most of
    # it does NOT resolve cleanly (multiple no-matches, multiple weak/partial
    # hits, or items we couldn't even read), rushing to mention the one or two
    # matches we scraped is worse than useless: it looks like the bot ignored
    # the list. The signal is retrieval quality (verdicts derived from RAG
    # scores) plus counts — not any word list. Rule: on a list, if the
    # unresolved items are at least two AND they are not outnumbered by clean
    # hits, STOP proposing and ask the client for codes / a clearer list.
    clean = len(resolved) + len(chosen)
    unresolved = len(missing) + len(ambiguous) + max(0, held_count)
    total = clean + unresolved
    batch_stop = total >= 3 and unresolved >= 2 and clean <= unresolved
    if batch_stop:
        digest = {
            "estado_lista": {
                "identificados_con_seguridad": clean,
                "sin_identificar_o_dudosos": unresolved,
                "total_detectado": total,
            },
            "instruccion_lista_ilegible": (
                "La lista llegó en su mayoría ilegible o ambigua: NO menciones las pocas "
                "coincidencias sueltas ni las apuntes (una coincidencia parcial es el último "
                "recurso, no el primero). Reconoce con naturalidad que has recibido la lista "
                "pero que así no puedes identificar los productos con seguridad y no quieres "
                "equivocarte. Pide amablemente que te la pase de forma más clara: MEJOR los "
                "códigos de producto (uno por línea), o si no, los nombres completos / marcas. "
                "NO vuelvas a buscar hasta que llegue esa lista. Una sola petición, cordial."
            ),
        }
        # Discard this turn's weak reads; keep only prior grounded carriers so
        # the client's clean re-send starts from a clean slate.
        persist_stop = [i for i in carried_queue if i.get("estado") == "propuesto"]
        seen_s: set = set()
        deduped_stop: List[Dict[str, Any]] = []
        for it in persist_stop:
            first = (it.get("opciones") or [{}])[0].get("codigo") or it.get("pedido") or ""
            key = (it.get("estado"), _sane_code(first) or first.lower())
            if key not in seen_s:
                seen_s.add(key)
                deduped_stop.append(it)
        logger.info(f"[AGENT] Batch-quality STOP: clean={clean} unresolved={unresolved} "
                    f"total={total}; asking for codes/clean list, not proposing partials.")
        return digest, deduped_stop
    # Previously queued items go first: the client is mid-list. Only items
    # that still need a client decision (enrichment or option-choice) belong
    # in the ask queue; carried "propuesto" grounding carriers are passed
    # through untouched and never re-asked.
    carried_ask = [i for i in carried_queue if i.get("estado") != "propuesto"]
    carried_grounded = [i for i in carried_queue if i.get("estado") == "propuesto"]
    queue = list(carried_ask) + ambiguous
    ask_now = queue[0] if queue else None
    remaining = queue[1:]
    # GROUNDING PERSISTENCE (breaks the re-propose loop): resolved/leading
    # hits were catalog-validated by THIS search, but the model may propose
    # them ("¿te apunto?") instead of adding them. If we don't persist them,
    # next turn's "sí" has no grounded code to confirm -> add_item defers ->
    # re-search -> the same "resolved" digest -> the model proposes again.
    # We carry each grounded code as estado="propuesto" so it lands in the
    # next turn's seen_codes; add_item's own cleanup drops it the moment the
    # code is actually added, so same-turn adds don't linger.
    grounded_proposals = carried_grounded + [
        {"pedido": r["pedido"], "estado": "propuesto",
         "opciones": [{"codigo": r["codigo"], "nombre": r.get("nombre", "")}]}
        for r in resolved + chosen if r.get("codigo")
    ]
    persist = queue + grounded_proposals
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
    # dedupe persisted items by (estado, first-code/pedido) so a code that is
    # both freshly resolved and already carried doesn't double up.
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for it in persist:
        first = (it.get("opciones") or [{}])[0].get("codigo") or it.get("pedido") or ""
        key = (it.get("estado"), _sane_code(first) or first.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)
    return digest, deduped


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

    def __init__(self, session: Dict[str, Any], media_text: str = "") -> None:
        # Media turns (image/audio/doc transcriptions) may be machine
        # misreadings: adds are allowed ONLY for codes literally present in
        # the extraction (dictated codes); anything else must be proposed
        # and confirmed. Alphanumeric-normalized containment check.
        self._media_alnum: Optional[str] = (
            re.sub(r"[^A-Z0-9]", "", media_text.upper()) if media_text else None)
        self.blocked_media_adds: List[str] = []
        self.blocked_media_close: bool = False
        self._close_requested: bool = False
        # Multi-item work queue persisted across turns; option codes are
        # catalog-verified from a prior search -> grounded for proposals/adds.
        self.open_items: List[Dict[str, Any]] = list(session.get("open_items") or [])
        self._guide_shown_out: bool = bool(session.get("guide_shown"))
        self.held_for_enrichment: List[str] = []
        self.held_qty: Dict[str, int] = {}   # lowercase held query -> client qty (triage)
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
                # Queries are collected RAW here; the loop gate routes them
                # (Pass-1 classes, query-gate helper, topic grounding). The
                # executor never judges query semantics.
                for q in (args.get("queries") or []):
                    q = str(q or "").strip()
                    if q and q not in queries:
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
                # Decide AFTER the whole tool pass (below), so the order of
                # add_item vs close_order in this turn's calls doesn't matter.
                self._close_requested = True
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
        # Resolve a requested close now that every add_item in this turn has
        # been processed. On a MEDIA turn the close is blocked ONLY when the
        # same audio/image also tried to add items that couldn't be verified
        # (blocked_media_adds) — that is the dangerous "add X and close" case
        # that would ship an incomplete cart. A plain voice confirmation of an
        # already-built cart ("ciérralo así, gracias") is honored: the cart was
        # confirmed in earlier text turns, nothing is pending.
        if getattr(self, "_close_requested", False):
            if not self.cart:
                self.blocked_empty_close = True
                logger.warning("[AGENT] close_order ignored: empty cart.")
            elif self._media_alnum is not None and self.blocked_media_adds:
                self.blocked_media_close = True
                logger.warning("[AGENT] close_order BLOCKED: media turn with unverified adds "
                               "pending (confirm those as text first).")
            else:
                self.status = "CLOSED"
                logger.info(f"[AGENT] Order CLOSED via close_order ({len(self.cart)} items).")
            self._close_requested = False
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
    if "<NO_REPLY>" in text:
        return ""
    # Small models sometimes ACKNOWLEDGE our internal nudges in the client
    # reply ("...basándome en esa instrucción interna", "interpreto que me
    # pides que reformule..."). Those references are to OUR machinery, never
    # to anything the client wrote — drop any sentence that mentions the
    # internal-instruction marker or narrates a reformulation request. This is
    # structural (our own marker words), not client-intent vocabulary.
    parts = re.split(r"(?<=[.!?\n])\s+", text)
    kept = [p for p in parts if not re.search(
        r"instrucci[oó]n interna|reformul\w+ mi respuesta|"
        r"me est[aá]s pidiendo que (reformul|reescrib|reh[ai]c)",
        _strip_accents_lower(p).replace("ó", "o"))]
    cleaned = " ".join(kept).strip()
    if cleaned != text:
        logger.info("[AGENT] Stripped a leaked internal-instruction reference from the reply.")
    return cleaned


def _fallback_reply(cart: List[Dict[str, Any]], open_items: List[Dict[str, Any]],
                    client_name: str, salesman_name: str, greet: bool) -> str:
    """Deterministic confirmation built from state — the last resort when the
    model spends its whole step budget without ever writing text. Never empty:
    a client who just added items and asked for a summary must get an answer,
    not silence."""
    parts: List[str] = []
    if greet:
        parts.append(f"¡Hola {client_name}! Soy Kapa, el asistente de {salesman_name}.")
    lines = [f"{i.get('qty', 1)}x código de producto {i['code']}"
             for i in (cart or []) if i.get("code")]
    if lines:
        parts.append("Te confirmo lo que tenemos apuntado: " + "; ".join(lines) + ".")
    else:
        parts.append("Todavía no tengo nada apuntado en el pedido.")
    pend = [i.get("pedido") for i in (open_items or [])
            if i.get("estado") == "enriquecer" and i.get("pedido")]
    if pend:
        parts.append("Me falta un poco más de detalle para " + ", ".join(pend) +
                     " (marca, línea o medida); el código de producto es lo más rápido y seguro.")
    parts.append("¿Te apunto algo más o cerramos el pedido?")
    return " ".join(parts)


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
    One conversation turn as a TWO-PASS orchestration:
      Pass 1  run_triage() at temperature 0.0 -> escalation short-circuit,
              WORTHY/AMBIGUOUS classification (gates retrieval below).
      Pass 2  LLM <-> tools loop (max MAX_STEPS calls), grounded live
              searches, code-validated state edits.
    Blocking cima calls run in a thread so the event loop stays free.
    """
    # MULTI-ACTOR SILENCE GUARD: if the human salesman (Comercial:) was the
    # last to speak, he has taken manual control of the conversation — the
    # bot stays completely silent. The FSM normally cancels the trigger on
    # a manual reply, but a salesman message landing DURING processing (or a
    # missed on_user_message) must not be answered over.
    last_line = next((l for l in reversed((recent_history or "").splitlines()) if l.strip()), "")
    if last_line.startswith("Comercial:"):
        logger.info("[AGENT] Salesman spoke last (manual control); staying silent.")
        return AgentResult(
            reply="", order_status=session.get("order_status") or "IDLE",
            cart=list(session.get("cart") or []), summary=session.get("summary") or "",
            open_items=list(session.get("open_items") or []),
            guide_shown=bool(session.get("guide_shown")), silent=True,
        )

    # ---- PASS 1: intent & specificity triage (temperature 0.0) -------------
    triage = await run_triage(current_message, recent_history)

    escalation_hint = False
    if triage.ok and triage.escalation:
        # Honor the escalation immediately ONLY when the helper's VERBATIM
        # evidence quote actually appears in the client's message — a
        # structural containment check, not a keyword list. A 2B helper can
        # misfire or paraphrase; an unbacked flag becomes a hint the main
        # agent sees (it can still call handoff_to_human, with its own
        # demote-once gate).
        norm_msg = _strip_accents_lower(current_message or "")
        corroborated = bool(triage.evidence) and \
            _strip_accents_lower(triage.evidence) in norm_msg
        if corroborated:
            # Strict <NO_REPLY> semantics: completely silent handoff — no
            # canned message, no LLM turn — so the salesman takes over clean.
            logger.info(f"[AGENT] Pass-1 {TRIAGE_ESC} corroborated "
                        f"(evidence={triage.evidence!r}); silent handoff.")
            return AgentResult(
                reply="", order_status=session.get("order_status") or "IDLE",
                cart=list(session.get("cart") or []), summary=session.get("summary") or "",
                open_items=list(session.get("open_items") or []),
                guide_shown=bool(session.get("guide_shown")),
                handoff=True, silent=True,
                ctx={"triage_ms": round(triage.elapsed_ms, 1), "triage_escalation": True},
            )
        escalation_hint = True
        logger.info("[AGENT] Pass-1 escalation flag NOT evidence-backed; passing as a hint.")

    # PURELY NON-COMMERCIAL small talk (and nothing else) -> stay silent. Kapa
    # is an ORDER assistant, not a chit-chat / general-knowledge bot: only
    # genuinely off-topic personal talk ("¿qué tal tu madre?", "¿capital de
    # Francia?") gets no reply. A GREETING or order-opener ("hola", "quiero
    # hacer un pedido") is NEVER silence — the client is very likely about to
    # order — so `greeting` vetoes the silence, as does any order item or
    # commercial flag. When the 2B triage is unsure it errs toward answering.
    if triage.ok and triage.small_talk and not triage.greeting \
            and not triage.items and not triage.refer_salesman \
            and not triage.repeat_order and not triage.escalation and not triage.opt_out:
        logger.info("[AGENT] Pass-1 charla_no_comercial (no greeting/order/commercial); staying silent.")
        return AgentResult(
            reply="", order_status=session.get("order_status") or "IDLE",
            cart=list(session.get("cart") or []), summary=session.get("summary") or "",
            open_items=list(session.get("open_items") or []),
            guide_shown=bool(session.get("guide_shown")), silent=True,
            ctx={"triage_ms": round(triage.elapsed_ms, 1), "triage_small_talk": True},
        )

    topic = _topic_from_history(recent_history)   # structural product-family grounding
    executor = _ToolExecutor(session,
                             media_text=current_message if _is_media_message(current_message) else "")

    # QUEUE RECONCILIATION (extra LLM pass, temp 0.0): read the running queue
    # against the new message and apply what the CLIENT decided — drop
    # abandoned items, commit confirmations of already-located ones. This is
    # what keeps the thread coherent turn-to-turn; without it the bot re-asks
    # items the client dropped and fails to honor a plain "sí". The model
    # decides; the code below only applies the verdict.
    reconciled_msg = ""
    if executor.open_items:
        rec = await run_open_items_reconcile(current_message, executor.open_items)
        if rec:
            dropped, confirmed = [], []
            if rec["descartar"]:
                keep = []
                for it in executor.open_items:
                    ped = it.get("pedido", "")
                    if any(_same_item(ped, d) for d in rec["descartar"]):
                        dropped.append(ped)
                    else:
                        keep.append(it)
                executor.open_items = keep
            for c in rec["confirmar"]:
                code = c["codigo"]
                # Confirmations only commit codes already grounded (proposed
                # earlier or in the cart/last order) — never an invented code.
                if code and code in executor.seen_codes:
                    executor.cart[code] = {"code": code, "qty": c["cantidad"]}
                    executor.order_activity = True
                    executor.status = "BUILDING" if executor.status != "CLOSED" else executor.status
                    executor.open_items = [
                        i for i in executor.open_items
                        if code not in {_sane_code(o.get("codigo")) for o in i.get("opciones", [])}
                    ]
                    confirmed.append(f"{c['cantidad']}x {code}")
            notes = []
            if dropped:
                notes.append("el cliente DESCARTÓ (no los menciones más): " + ", ".join(dropped))
            if confirmed:
                notes.append("ya APUNTADO por confirmación del cliente (dilo como hecho, no "
                             "vuelvas a preguntar '¿te apunto?'): " + ", ".join(confirmed))
            if dropped or confirmed:
                reconciled_msg = "### YA APLICADO ESTE TURNO\n- " + "\n- ".join(notes) + "\n"
                logger.info(f"[RECONCILE] applied dropped={dropped} confirmed={confirmed}")

    # AMBIGUOUS items are queued for enrichment UP FRONT (with the client's
    # quantities): even if the agent never emits a query for them, they land
    # in open_items at _finish and survive across turns.
    for item in (triage.ambiguous if triage.ok else []):
        if not any(_same_item(item.query, h) for h in executor.held_for_enrichment) and \
                not any(_same_item(item.query, i.get("pedido", "")) for i in executor.open_items):
            executor.held_for_enrichment.append(item.query)
            executor.held_qty[item.query.strip().lower()] = item.qty

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
    triage_block = _triage_block(triage, escalation_hint)
    if reconciled_msg:
        triage_block = reconciled_msg + triage_block
    # PROACTIVE media / big-list guidance (prepended to the context block, not
    # a reactive nudge round the 2B model can echo). A photo of a handwritten
    # list or a long dictated order is a core use case: the natural move is to
    # say plainly what came through and what didn't, group the matches, and
    # ask for codes on the unclear ones — one message, not an interrogation.
    total_mentions = (len(triage.items) if triage.ok else 0) + len(executor.open_items)
    if _is_media_message(current_message):
        triage_block = (
            "### ESTO LLEGÓ COMO IMAGEN/AUDIO (interpretación, puede fallar)\n"
            "- El texto trae al inicio una DESCRIPCIÓN de qué es la imagen y lo legible que "
            "es. Úsala: si dice que es una lista manuscrita difícil, borrosa o casi ilegible, "
            "díselo con naturalidad al cliente ('me llega como una lista escrita a mano y me "
            "cuesta leerla bien') y pídele una foto más clara o, mejor, los códigos de producto.\n"
            "- Reconoce con naturalidad que te llegó por imagen/audio y que has leído lo "
            "que has podido; NO te disculpes en exceso ni hables de instrucciones internas.\n"
            "- Di qué has entendido y qué NO; para lo dudoso pide con humildad el código de "
            "producto o el nombre exacto. Invita SIEMPRE a corregirte.\n"
            "- Nunca apuntes nada de una imagen sin que el cliente confirme.\n"
        ) + triage_block
    elif total_mentions >= 5:
        triage_block = (
            "### LISTA LARGA — sé natural, no interrogues\n"
            "- Empieza reconociendo el pedido COMPLETO en una frase (qué tienes claro, qué "
            "tiene opciones, a qué le falta detalle).\n"
            "- Agrupa: confirma de golpe lo que localizaste; para lo genérico o dudoso pide "
            "el código de producto o el nombre exacto. UNA sola tanda de preguntas, no una "
            "por artículo.\n"
        ) + triage_block
    elif not _is_media_message(current_message) and _should_remind_codes(recent_history) \
            and (executor.cart or executor.open_items or (triage.ok and triage.items)):
        # From time to time on a live order, drop the codes tip once — not
        # every turn (the suppression window handles the cadence).
        triage_block = (
            "### RECORDATORIO SUAVE (solo si encaja natural)\n"
            "- Recuérdale de pasada, sin insistir y en una frase, que el código de producto "
            "es la forma más rápida y segura de asegurar los artículos sin errores.\n"
        ) + triage_block
    open_line = "; ".join(
        (f"{i.get('pedido')} (x{i.get('qty', 1)}, falta detalle del cliente)"
         if i.get("estado") == "enriquecer" else
         (f"{i.get('pedido')} → YA localizado {(i.get('opciones') or [{}])[0].get('codigo','')} "
          f"(si el cliente confirma, add_item directo; NO lo vuelvas a buscar)"
          if i.get("estado") == "propuesto" else
          f"{i.get('pedido')} (opciones: " + ", ".join(
              o.get("codigo", "") for o in i.get("opciones", [])[:3]) + ")"))
        for i in executor.open_items) or "(ninguno)"
    shell = _USER.format(
        order_status=executor.status,
        open_items_line=open_line,
        cart_json=json.dumps(list(executor.cart.values()), ensure_ascii=False),
        last_closed_json=json.dumps(session.get("last_closed_cart") or [], ensure_ascii=False),
        summary=executor.summary or "(sin memoria previa)",
        history="",
        triage_block=triage_block,
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
            triage_block=triage_block,
            current_message=current_message or "(sin texto)",
        )),
    ]

    # ---- context telemetry (est. tokens; the gauge the console monitors) ----
    ctx_stats: Dict[str, Any] = {
        "window": CTX_WINDOW, "budget": CTX_BUDGET_TOKENS, "used": 0,
        "history_chars_kept": len(history), "history_chars_total": len(recent_history or ""),
        "message_truncated": "[recortado]" in (current_message or ""),
        "triage_ok": triage.ok, "triage_ms": round(triage.elapsed_ms, 1),
        "triage_worthy": len(triage.worthy), "triage_ambiguous": len(triage.ambiguous),
        "triage_escalation_hint": escalation_hint,
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
            # SIGNATURE: guarantee the client can tell this is the assistant,
            # not the human salesman — on EVERY delivered message, regardless
            # of intro mode. Deterministic and idempotent (skip if the exact
            # signature is already the tail, e.g. from a retried draft).
            sig = BOT_SIGNATURE.strip()
            if sig and _strip_accents_lower(sig) not in _strip_accents_lower(rep[-len(sig) - 4:]):
                rep = rep.rstrip() + "\n\n" + sig
        existing = {i.get("pedido") for i in executor.open_items}
        for h in executor.held_for_enrichment:
            key = h.strip().lower()
            if key and key not in existing:
                executor.open_items.append({"pedido": key, "estado": "enriquecer",
                                            "opciones": [],
                                            "qty": executor.held_qty.get(key, 1)})
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
    audit_calls = 0     # reply-audit helper budget per turn
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

        # Escalations: honor when the Pass-1 triage's independent read of the
        # message corroborates them (rechazo_bot / derivacion / escalada),
        # when the model insists (second call), or when no nudge round remains.
        can_nudge = step < budget - 1
        if executor.opt_out and can_nudge and not demoted_optout and \
                not (triage.ok and (triage.opt_out or triage.escalation)):
            executor.opt_out = False
            demoted_optout = True
            logger.info("[AGENT] opt_out demoted (Pass-1 saw no bot rejection in the message).")
            messages.append(Message(role="user", content=_internal(
                "No consta que el cliente rechace hablar con un asistente. Responde tú mismo a su "
                "mensaje (o llama a opt_out_client otra vez SOLO si realmente no quiere hablar contigo).")))
            continue
        if executor.handoff and can_nudge and not demoted_handoff and \
                not (triage.ok and triage.escalation):
            # Honor handoff only when Pass-1 saw the client actually want a
            # person. A derivacion (price/stock/incident question) must NOT
            # become a handoff — the bot SPEAKS the referral and stays active.
            executor.handoff = False
            demoted_handoff = True
            reason = ("El cliente pregunta algo comercial que no puedes resolver: NO uses "
                      "handoff_to_human. Dile en una frase, amable, que eso se lo confirma "
                      f"{salesman_name}, y sigue tú con lo que sí puedas (el pedido).") \
                if (triage.ok and triage.refer_salesman) else \
                (f"No consta que el cliente pida hablar con {salesman_name}. Responde tú "
                 "mismo siguiendo tus reglas (o llama a handoff_to_human otra vez SOLO si de "
                 "verdad no puedes ayudarle).")
            logger.info("[AGENT] handoff demoted (Pass-1: not a request for a person).")
            messages.append(Message(role="user", content=_internal(reason)))
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
        if executor.opt_out:
            return await _finish("")  # opt-out: caller sends the brief stop message
        if executor.handoff:
            # A MODEL-initiated handoff that survived the demote. We never emit
            # the robotic canned redirect for this — a redirect must be natural
            # and lead with what the bot CAN do. (Silence for a genuine "wants
            # a person" is the Pass-1 corroborated-escalation path, handled
            # before the loop.) Use the model's own reply if it wrote one;
            # otherwise force ONE capability-explaining reply, no tools.
            executor.handoff = False
            if reply.strip():
                return await _finish(reply)
            messages.append(Message(role="user", content=_internal(
                "No uses ningún mensaje enlatado de derivación. Responde TÚ, natural y breve: "
                "di primero qué SÍ puedes hacer (localizar productos por nombre completo o "
                "código de producto y montar el pedido); y SOLO para lo que de verdad no te "
                f"toca (precios, stock, facturas, incidencias, o hablar con una persona) di que "
                f"eso se lo confirma {salesman_name}. Termina invitándole a decirte qué producto "
                "busca. NO llames a ninguna herramienta.")))
            try:
                resp = await asyncio.to_thread(
                    llm.chat, CIMA_MODEL, messages,
                    options={"temperature": AGENT_TEMPERATURE, "top_p": 0.9,
                             "repeat_penalty": 1.1, "num_predict": 400})
                forced = _clean(resp.content)
            except Exception as e:
                logger.error(f"[AGENT] Forced capability reply failed: {e}")
                forced = ""
            return await _finish(forced)
        # A demoted mixed turn falls through as a normal order turn: the reply
        # still passes the grounding/humility/parroting guards. If the model
        # insists on handoff_to_human in the NEXT round, it is honored.

        # REPLY AUDIT (temperature-0.0 helper, at most twice per turn): judges
        # the draft semantically — announced-but-not-called searches, and the
        # humility invitation on media-derived proposals. No phrase regexes.
        needs_promise_check = reply and not queries and can_nudge and not demoted_promise
        needs_humility_check = (reply and not queries and can_nudge and not demoted_humility
                                and _is_media_message(current_message)
                                and bool(_proposal_codes(reply)))
        audit: Optional[Dict[str, bool]] = None
        if (needs_promise_check or needs_humility_check) and audit_calls < 2:
            audit_calls += 1
            ctx_stats["helper_calls"] = ctx_stats.get("helper_calls", 0) + 1
            audit = await run_reply_audit(reply)

        if needs_promise_check and audit and audit.get("promete_busqueda"):
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

        if needs_humility_check and audit and not audit.get("invita_correccion"):
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
            if triage.ok and triage.repeat_order:
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
            uniq = list(dict.fromkeys(queries))
            route: Dict[str, str] = {}          # query -> RUN | HOLD | ATTR
            uncovered: List[str] = []
            for q in uniq:
                ql = q.strip().lower()
                if ql in enrich_pending or any(_same_item(q, p or "") for p in enrich_pending):
                    route[q] = "RUN"            # client already engaged on this item
                elif _code_shaped(q):
                    route[q] = "RUN"            # a code is always searchable, turn one included
                else:
                    cls = _classify_query(q, triage)
                    if cls == TRIAGE_WORTHY:
                        route[q] = "RUN"        # Pass-1 verdict: specific enough — search NOW
                    elif cls == TRIAGE_AMBIGUOUS:
                        route[q] = "HOLD"       # Pass-1 verdict: needs conversational narrowing
                    elif cls == TRIAGE_ATTRIBUTE:
                        route[q] = "ATTR"       # Pass-1 verdict: variant of the current topic
                    else:
                        uncovered.append(q)     # invented mid-loop: ask the query-gate helper
            if uncovered:
                ctx_stats["helper_calls"] = ctx_stats.get("helper_calls", 0) + 1
                verdicts = await run_query_gate(uncovered)
                for q in uncovered:
                    # Missing verdict -> PERMISSIVE (run it): the result triage
                    # and the uncovered-token honesty rule absorb noisy hits.
                    v = verdicts.get(q.strip().lower(), "SEARCHABLE")
                    route[q] = {"SEARCHABLE": "RUN", "GENERIC": "HOLD", "ATTRIBUTE": "ATTR"}[v]
            runnable, held = [], []
            for q in uniq:
                verdict = route[q]
                if verdict == "ATTR":
                    if topic:
                        grounded = f"{' '.join(topic)} {q}"
                        logger.info(f"[AGENT] Attribute query grounded with topic: '{q}' -> '{grounded}'")
                        runnable.append(grounded)
                    else:
                        held.append(q)          # no product context: ask which product it refers to
                elif verdict == "HOLD":
                    held.append(q)
                else:
                    runnable.append(q)
            if held:
                logger.info(f"[AGENT] Queries HELD for enrichment (no Qdrant call): {held}")
                for h in held:
                    if not any(_same_item(h, e) for e in executor.held_for_enrichment):
                        executor.held_for_enrichment.append(h)
                        executor.held_qty.setdefault(h.strip().lower(), _triage_qty(h, triage))
            if not runnable:
                if can_nudge and not demoted_generic_q:
                    demoted_generic_q = True
                    messages.append(Message(role="assistant", content=json.dumps(
                        {"tool_calls": [{"search_products": queries}]}, ensure_ascii=False)))
                    held_line = "; ".join(
                        f"{h} (x{executor.held_qty.get(h.strip().lower(), 1)})" for h in held)
                    nudge = (
                        f"Estas menciones son demasiado genéricas para identificar un artículo "
                        f"entre miles ({held_line}): NO las busques a ciegas. Si el mensaje o el "
                        f"historial dan más contexto (marca, línea, medida, color), busca con ESO. "
                        f"Si no, resume con naturalidad lo que has entendido (con sus cantidades) "
                        f"y pide amablemente un poco más de detalle, recordando con humildad que "
                        f"el código de producto es lo más rápido y seguro para apuntarlo sin "
                        f"error; si dice que se llama así tal cual, lo buscarás. Una sola pregunta.")
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
                is_media=_is_media_message(current_message),
                held_count=len(executor.held_for_enrichment))
            batch_stopped = "instruccion_lista_ilegible" in digest
            if batch_stopped:
                # The list was mostly unreadable: we asked for codes/a clean
                # list. Drop this turn's held reads so they don't reappear as
                # per-item questions and muddy the single cordial request.
                executor.held_for_enrichment.clear()
            elif executor.held_for_enrichment:
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

    # Budget spent with NO reply on a turn that did real work: the small model
    # burned its steps tool-calling (observed live: re-searching an
    # already-searched code, a promise-nudge eating a round) and never wrote
    # text. Silence here strands a client who is mid-order and even asked for a
    # summary. Force ONE text-only call (no tools -> it MUST answer); if that
    # still yields nothing, send a deterministic confirmation from state.
    # Genuine silence (handoff / opt-out / nothing happened) is left untouched.
    active_turn = (executor.order_activity or executor.results_this_turn
                   or executor.cart or executor.open_items or executor.pending_adds)
    if not reply and not executor.handoff and not executor.opt_out and active_turn:
        logger.info("[AGENT] Budget spent with no reply on an active turn; forcing a final answer.")
        messages.append(Message(role="user", content=_internal(
            "Se acabaron los pasos. Escribe AHORA la respuesta final al cliente siguiendo tus "
            "reglas: confirma con naturalidad lo que has apuntado (con sus códigos y cantidades), "
            "y si pidió un resumen, dáselo a partir del carrito actual. NO llames a ninguna "
            "herramienta; responde solo con el texto para el cliente.")))
        try:
            resp = await asyncio.to_thread(
                llm.chat, CIMA_MODEL, messages,   # NO tools: text is the only valid output
                options={"temperature": AGENT_TEMPERATURE, "top_p": 0.9,
                         "repeat_penalty": 1.1, "num_predict": 512},
            )
            reply = _clean(resp.content)
        except Exception as e:
            logger.error(f"[AGENT] Forced final-answer call failed: {e}")
        if not reply:
            greet = intro_mode == "new" or (
                intro_mode == "renew" and (executor.order_activity or executor.results_this_turn))
            reply = _fallback_reply(list(executor.cart.values()), executor.open_items,
                                    client_name, salesman_name, greet)
            logger.warning("[AGENT] Model still silent; using deterministic fallback reply.")

    reply, leaked_note = _split_leaked_note(reply)
    if leaked_note and not executor.summary_touched:
        executor.summary = leaked_note[:300]
        logger.info(f"[AGENT] Leaked visible note rerouted to summary: '{leaked_note[:60]}'")
    return await _finish(reply)