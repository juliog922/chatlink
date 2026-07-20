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


def _messages_tokens(msgs: List["Message"]) -> int:
    """Estimated tokens for the whole message list actually sent to the model,
    including per-message chat-template overhead."""
    return sum(_est_tokens(m.content or "") + 4 for m in msgs)


def _fit_messages(msgs: List["Message"], budget: int) -> List["Message"]:
    """Guarantee the assembled context stays under `budget` tokens even after
    the loop has appended tool results and nudges. The system prompt (msgs[0]),
    the main state/context turn (msgs[1]) and the most recent turn are always
    kept; the OLDEST middle chatter (stale tool results, superseded tool-call
    announcements, spent nudges) is dropped first — its information already
    lives in the cart/open_items state and the persisted note summary."""
    if len(msgs) <= 3 or _messages_tokens(msgs) <= budget:
        return msgs
    head, middle, tail = msgs[:2], msgs[2:-1], msgs[-1:]
    while middle and _messages_tokens(head + middle + tail) > budget:
        middle.pop(0)
    return head + middle + tail

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
Eres Kapa, el asistente de pedidos de {salesman_name} (cosmética). Atiendes a \
sus clientes por WhatsApp cuando él no está. Hablas natural y breve, en el \
idioma del cliente, con algún emoji; no mencionas sistemas ni IA. {intro_rule}

LO QUE SÍ HACES: localizar productos del catálogo (search_products, por nombre \
o código de producto) y montar el pedido. Di siempre "código de producto".

LO QUE NO HACES (lo lleva {salesman_name}): mostrar el catálogo entero, dar \
precios, stock, facturas, incidencias/devoluciones, envíos o consejo de \
producto. Ante eso, dilo con naturalidad ("eso te lo confirma {salesman_name}") \
y sigue con lo que sí puedas. No hace falta derivar por esto.

CÓMO TRABAJAS:
- Busca solo PRODUCTOS concretos (nombre o código). Una marca o categoría sola \
("gelfix", "una crema") no basta para buscar: el cliente tiene el catálogo, \
así que pídele el producto concreto o el código. Nunca digas "un momento / voy \
a mirar": busca ya o responde con lo que tienes.
- Solo existen los productos que devuelve la búsqueda: NUNCA inventes códigos, \
nombres ni marcas. Muestra solo lo que encaje; si no encaja, dilo y pide \
detalle o el código.
- Confirma antes de apuntar: con el "sí" del cliente usas add_item. El carrito \
se mantiene entre mensajes. Al terminar ("nada más", "ciérralo", "envíalo"): \
resume el carrito y usa close_order, pasándole la nota a {salesman_name}.
- El código de producto es lo más rápido y seguro; recuérdalo de vez en \
cuando, sin insistir.

LA CONVERSACIÓN (tres voces: "Cliente:", "Comercial:" = {salesman_name} en \
persona, "Asistente:" = tú): léela entera. Si el Comercial ya respondió, no lo \
repitas ni lo contradigas; si dejó un pedido a medias y el cliente sigue, \
continúa desde ahí. No conoces sus charlas fuera del chat: si el cliente se \
refiere a algo que no ves, pídele el nombre o el código.

DECIDE TÚ, con naturalidad, cómo responder a cada mensaje. Usa handoff_to_human \
SOLO si el cliente pide hablar con una persona o de verdad no hay nada que \
puedas hacer (entonces te retiras y entra {salesman_name}). Usa opt_out_client \
si rechaza hablar con un asistente. Si el mensaje es puramente ajeno al negocio \
y no aporta nada al pedido, responde "<NO_REPLY>".

IMÁGENES/AUDIOS: son interpretaciones y puedes fallar. Di lo que entendiste, \
invita a corregir, y NUNCA apuntes ni cierres sin que el cliente confirme.

Las [INSTRUCCIÓN INTERNA] no son del cliente: cúmplelas sin citarlas ni \
disculparte por ellas."""

_INTRO_RULES: Dict[str, str] = {
    "new": ("CONTACTO NUEVO: preséntate en UNA frase (Kapa, asistente de {salesman_name}) y "
            "EN LA MISMA respuesta atiende lo que haya pedido. Solo si únicamente saluda, "
            "pregúntale con tus palabras qué necesita."),
    "renew": ("La conversación anterior TERMINÓ. Si empieza una gestión de pedido, preséntate "
              "breve de nuevo antes de atenderle; si solo saluda o se despide, responde cordial "
              "SIN presentarte."),
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
    ok: bool = False                 # False -> helper failed; agent extracts queries itself
    repeat_order: bool = False       # asks to repeat the usual / last order (grounding hint only)
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
Extraes los PRODUCTOS que el cliente quiere pedir de su mensaje. NO conversas: \
devuelves SOLO un objeto JSON válido con esta forma:
{"repetir_ultimo": false, "articulos": [{"mencion": "…", "consulta": "…", \
"cantidad": 1, "clase": "WORTHY"}]}

- "articulos": una entrada por producto que el cliente pide. [] si el mensaje \
no pide productos (saludos, dudas, precios, cierre, charla… no son artículos).
  - "mencion": el fragmento del cliente, con su cantidad si la dice.
  - "consulta": términos de búsqueda cortos (producto + marca/atributo/medida). \
Sin cantidades, sin verbos, sin la frase entera.
  - "cantidad": el número pedido (1 si no lo dice).
  - "clase":
    · "WORTHY" = hay un PRODUCTO buscable: tipo/nombre con marca, medida, \
atributo de línea o código (KG001399, 14-1127). Una marca sola ("gelfix") NO \
es WORTHY.
    · "AMBIGUOUS" = genérico: categoría sola ("una crema"), marca sola sin \
producto, o algo demasiado vago para buscar sin ruido.
    · "ATTRIBUTE" = solo un atributo/variante del producto del que ya se \
hablaba ("el negro", "la grande", "la 2").
- "repetir_ultimo": true solo si pide repetir su pedido habitual/anterior \
("lo de siempre", "lo mismo que la última vez")."""

_TRIAGE_USER = """\
Historial reciente (solo contexto, NO lo clasifiques):
{history}

MENSAJE NUEVO DEL CLIENTE (extrae SOLO de esto):
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
        # Truncated output: reconstruct the repeat flag and every complete
        # {...} object already emitted inside the (unterminated) array.
        data = {}
        m = re.search(r'"repetir_ultimo"\s*:\s*(true|false)', text[start:])
        if m:
            data["repetir_ultimo"] = m.group(1) == "true"
        head = text[start:]
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
        if not objs and "repetir_ultimo" not in data:
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
    logger.info(f"[TRIAGE] ok in {elapsed:.0f}ms: repetir={parsed.repeat_order} "
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


def _triage_block(triage: TriageResult) -> str:
    """Pass-1 product extraction injected into Kapa's context — search guidance
    only. All intent handling (greet, close, refer, handoff, small talk) is the
    model's job now, driven by the brief system prompt and its tools."""
    if not triage.ok or (not triage.items and not triage.repeat_order):
        return ""
    lines: List[str] = ["### PRODUCTOS DETECTADOS EN EL MENSAJE (análisis interno; el cliente NO lo ve)"]
    if triage.worthy:
        lines.append("- Concretos, BUSCA YA con search_products (una consulta por artículo): " +
                     "; ".join(f'"{i.query}" (x{i.qty})' for i in triage.worthy))
    if triage.ambiguous:
        lines.append("- Genéricos o solo marca, NO los busques: " +
                     "; ".join(f'"{i.query}" (x{i.qty})' for i in triage.ambiguous) +
                     ". Resume lo que entendiste y pide el producto concreto o el código.")
    if triage.attributes:
        lines.append("- Variantes del producto en curso (búscalas COMBINADAS con el producto "
                     "del que hablabais): " +
                     "; ".join(f'"{i.query}" (x{i.qty})' for i in triage.attributes))
    if triage.repeat_order:
        lines.append("- Pide REPETIR su pedido habitual: usa el 'Último pedido ENVIADO' del "
                     "contexto como base (add_item con esos códigos).")
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
- "SEARCHABLE": nombra un PRODUCTO concreto (tipo de producto o su nombre \
distintivo), a veces con marca/línea/medida, o un código: "crema kerapro", \
"delineador katai", "mascarilla antiedad 8gr", "14-1127". La clave es que hay \
un PRODUCTO, no solo una marca o una categoría.
- "GENERIC": NO identifica un producto: solo una categoría ("crema", \
"esmaltes", "champú"), solo una MARCA sin producto ("gelfix", "lamel", "de \
nivea"), preguntar qué hay de una marca o en general ("qué productos tienes", \
"qué tienes de gelfix", "el catálogo"), o palabras vacías ("algo barato"). \
Buscar esto devolvería cientos de resultados o ninguno útil: hay que pedir al \
cliente el producto o el código.
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

_MATCH_SYSTEM = """\
Decides, para cada producto que pidió el cliente, qué candidatos del catálogo \
son de verdad ESE producto. NO conversas: devuelves SOLO JSON:
{"resultados": [{"pedido": "…", "codigos": ["…"]}]}

Para cada "pedido" te doy candidatos del catálogo (código, nombre, marca). En \
"codigos" pon los códigos de los candidatos que SON ese producto (mismo tipo, \
marca y variante que pidió el cliente), del más probable al menos probable:
- Si uno coincide claramente, pon solo ese.
- Si varios encajan y no puedes distinguirlos con lo que dijo el cliente, \
ponlos todos (para que el cliente elija).
- Si NINGÚN candidato es ese producto, "codigos": [].
Usa SOLO códigos de los candidatos que te doy; nunca inventes. Un nombre \
parecido pero de otra variante/producto NO coincide."""


async def run_match(items: List[Dict[str, Any]]) -> Optional[Dict[str, List[str]]]:
    """One simple 0.0 call that replaces brittle score thresholds: given each
    request and its catalog candidates, return which candidate codes genuinely
    match. -> {pedido: [codigo…ranked]} or None on failure (callers then fall
    back to the score-based verdict, staying permissive)."""
    payload_items = [it for it in items if it.get("candidatos")]
    if not payload_items:
        return None
    payload = json.dumps({"items": payload_items}, ensure_ascii=False)
    try:
        resp = await asyncio.to_thread(
            _get_client().chat, CIMA_MODEL,
            [Message(role="system", content=_MATCH_SYSTEM),
             Message(role="user", content=_cap_middle(payload, MAX_RESULTS_CHARS))],
            fmt="json",
            options={"temperature": TRIAGE_TEMPERATURE, "top_p": 1.0, "num_predict": TRIAGE_MAX_TOKENS},
            timeout=TRIAGE_TIMEOUT_S,
        )
        text = (resp.content or "").strip()
        start, end = text.find("{"), text.rfind("}")
        data = json.loads(text[start:end + 1]) if start != -1 and end > start else None
    except Exception as e:
        logger.warning(f"[MATCH] Helper failed ({e}); falling back to score verdicts.")
        return None
    if not isinstance(data, dict):
        return None
    out: Dict[str, List[str]] = {}
    for r in (data.get("resultados") or []):
        if isinstance(r, dict) and r.get("pedido") is not None:
            codes = [_sane_code(c) for c in (r.get("codigos") or []) if _sane_code(c)]
            out[str(r["pedido"])] = codes
    logger.info(f"[MATCH] {out}")
    return out


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
                    held_count: int = 0,
                    match_map: Optional[Dict[str, List[str]]] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
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
        # Prefer the semantic MATCH verdict (which real catalog hits are this
        # product) over brittle score thresholds. Reorder hits to the matched
        # codes; fall back to the score-based verdict when the match helper
        # didn't cover this query.
        matched_codes = match_map.get(q) if match_map else None
        if matched_codes is not None:
            by_code = {_sane_code(h.get("CodigoArticulo")): h for h in hits}
            matched_hits = [by_code[c] for c in matched_codes if c in by_code]
            if not matched_hits:
                missing.append(q)
                continue
            hits = matched_hits
            verdict = "resolved" if len(matched_hits) == 1 else "ambiguous"
            uncovered = []
        else:
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
    # Previously queued items go first: the client is mid-list. Only items
    # that still need a client decision (enrichment or option-choice) belong
    # in the ask queue; carried "propuesto" grounding carriers are passed
    # through untouched and never re-asked.
    carried_ask = [i for i in carried_queue if i.get("estado") != "propuesto"]
    carried_grounded = [i for i in carried_queue if i.get("estado") == "propuesto"]
    queue = list(carried_ask) + ambiguous
    # GROUNDING PERSISTENCE (breaks the re-propose loop): resolved/leading hits
    # persist as estado="propuesto" so next turn's "sí" has a grounded code.
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
    # ALWAYS commit / propose the FULL matches — never make the client hunt for
    # products the search already pinned down.
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

    # The UNRESOLVED set (needs the client): ambiguous options + not-found +
    # generic-held. How we present it scales with how many there are — the
    # count is the only threshold (the WHAT-matched decision was the run_match
    # AI call). >2 unresolved would make a tedious "pick one for each" wall, so
    # collapse it into a single ask for names/codes; <=2 stays helpful and
    # concrete (options / mention). Full matches above are committed either way.
    unresolved_names = [i["pedido"] for i in ambiguous] + list(missing)
    n_unresolved = len(ambiguous) + len(missing) + max(0, held_count)
    if n_unresolved > 2:
        digest["no_encontrados"] = unresolved_names
        digest["instruccion_no_encontrados"] = (
            "Son varios los que NO has podido identificar con seguridad. NO listes opciones "
            "de cada uno (sería un mensaje larguísimo): di en UNA frase, natural, cuáles no "
            "has localizado y pídele esos por nombre completo o código de producto. Recuerda "
            "que el código es lo más rápido y seguro.")
        # Persist the unresolved as enrichment items so that when the client
        # sends names/codes next turn, reconcile/search can match them — but do
        # NOT surface per-item options now.
        for it in ambiguous:
            if not any(_same_item(it["pedido"], q.get("pedido", "")) for q in queue if q is not it):
                it["estado"] = "enriquecer"
    else:
        # <=2 unresolved: the concrete, helpful per-item behavior.
        ask_now = queue[0] if queue else None
        remaining = queue[1:]
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


def _has_explicit_qty(qty: Any) -> bool:
    """True only when the model actually supplied a usable quantity. A missing/
    null/non-numeric qty on a re-add must NOT overwrite a real cart quantity."""
    if qty is None:
        return False
    try:
        int(qty)
        return True
    except Exception:
        return False


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
        # Authoritative quantities parsed from FREE JSON (triage / reconcile),
        # keyed by code. The inference server's tool-call grammar can truncate
        # an integer argument to a single digit (10->1, 12->1), so when we have
        # a code's quantity from a non-tool path we trust THAT over add_item's
        # qty argument.
        self.qty_hints: Dict[str, int] = {}
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
                    # Re-adding a code ALREADY in the cart without an explicit
                    # quantity must NOT reset it to 1 — the 2B model re-emits
                    # add_item on the close turn to 'confirm', and that silently
                    # wiped real quantities (cart showed 1, reply showed N, the
                    # Excel shipped 1). Keep the existing qty unless the model
                    # gives a new valid one; set_qty remains the way to change it.
                    raw_qty = args.get("qty")
                    if code in self.qty_hints:
                        # Authoritative quantity from triage/reconcile (free
                        # JSON) — beats the tool-call arg, which the grammar may
                        # have truncated to a single digit.
                        qty = self.qty_hints[code]
                    elif code in self.cart and not _has_explicit_qty(raw_qty):
                        qty = self.cart[code]["qty"]
                    else:
                        qty = _sane_qty(raw_qty)
                    self.cart[code] = {"code": code, "qty": qty}
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
                    raw_qty = args.get("qty")
                    # Preserve a known qty (hint, cart, or prior pending add) on
                    # a bare re-add; only a new explicit qty overrides it.
                    if code in self.qty_hints:
                        self.pending_adds[code] = self.qty_hints[code]
                    elif not _has_explicit_qty(raw_qty):
                        self.pending_adds[code] = (self.cart.get(code, {}).get("qty")
                                                   or self.pending_adds.get(code) or 1)
                    else:
                        self.pending_adds[code] = _sane_qty(raw_qty)
                    queries.append(code)
            elif name == "remove_item":
                self.cart.pop(_sane_code(args.get("code")), None)
            elif name == "set_qty":
                code = _sane_code(args.get("code"))
                if code in self.cart:
                    self.cart[code]["qty"] = self.qty_hints.get(code) or _sane_qty(args.get("qty"))
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


_KNOWN_TOOLS = ("search_products", "add_item", "remove_item", "set_qty",
                "close_order", "handoff_to_human", "opt_out_client", "note")


def _recover_text_tool_calls(content: str) -> List[Dict[str, Any]]:
    """Parse tool calls the model emitted as TEXT (malformed, not via the tool
    channel) and normalize them to the executor's {"function": {...}} shape, so
    a search/add the model 'said' in prose still executes instead of vanishing.
    Handles {"tool_calls":[{name:args}]}, {"function":{"name","arguments"}}, and
    a bare {name: args}. Best-effort: unparseable blocks are ignored."""
    if not content or "{" not in content:
        return []
    blocks, i, n = [], 0, len(content)
    while i < n:
        if content[i] == "{":
            depth, j = 0, i
            while j < n:
                if content[j] == "{":
                    depth += 1
                elif content[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if depth == 0:
                blocks.append(content[i:j + 1])
                i = j + 1
                continue
        i += 1
    out: List[Dict[str, Any]] = []
    for blk in blocks:
        try:
            data = json.loads(blk)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        calls = data.get("tool_calls")
        if not isinstance(calls, list):
            calls = [data]        # bare {name: args} or {"function": {...}}
        for c in calls:
            if not isinstance(c, dict):
                continue
            fn = c.get("function")
            if isinstance(fn, dict) and fn.get("name") in _KNOWN_TOOLS:
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                out.append({"function": {"name": fn["name"], "arguments": args or {}}})
                continue
            for k, v in c.items():
                if k in _KNOWN_TOOLS:
                    out.append({"function": {"name": k,
                                             "arguments": v if isinstance(v, dict) else {}}})
    return out


def _clean(text: str) -> str:
    """Strip reasoning tags, the silence token, raw tool-call JSON the model
    leaked as text, and sentences where the model leaks its OWN internal
    process into the client reply."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    if "<NO_REPLY>" in text:
        return ""
    # The 2B model sometimes emits a TOOL CALL as plain text instead of through
    # the tool channel — observed live: a whole {"tool_calls":[{"note":{...}}]}
    # blob shipped to the client. Strip any JSON object that carries "tool_calls"
    # or names one of our tools, plus stray ```json fences. What remains (if
    # anything) is the real prose; an all-JSON reply becomes empty -> the caller
    # forces a proper answer / deterministic fallback instead of leaking.
    text = re.sub(r"```(?:json)?|```", "", text)
    _tool_names = ("tool_calls", "search_products", "add_item", "remove_item",
                   "set_qty", "close_order", "handoff_to_human", "opt_out_client", "note")
    tool_re = re.compile(r'"(?:' + "|".join(_tool_names) + r')"')
    # Remove balanced {...} blocks that mention a tool name (repeat until stable).
    def _strip_tool_json(s: str) -> str:
        out, i, n = [], 0, len(s)
        while i < n:
            if s[i] == "{":
                depth, j = 0, i
                while j < n:
                    if s[j] == "{":
                        depth += 1
                    elif s[j] == "}":
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                block = s[i:j + 1]
                if depth == 0 and tool_re.search(block):
                    i = j + 1          # drop this tool-call block
                    continue
                out.append(s[i]); i += 1
            else:
                out.append(s[i]); i += 1
        return "".join(out)
    if tool_re.search(text):
        text = _strip_tool_json(text).strip()
        logger.info("[AGENT] Stripped leaked tool-call JSON from the reply.")
    # Small models leak their machinery into the reply in a few recurring ways,
    # all about the BOT's own process (never anything the client wrote):
    #   - acknowledging our internal nudges ("...esa instrucción interna",
    #     "interpreto que me pides que reformule");
    #   - apologizing for a technical delay / narrating that it is waiting on a
    #     search ("disculpa, estaba esperando la respuesta de la búsqueda");
    #   - promising to go look instead of having looked ("dame un segundo
    #     mientras lo reviso", "voy a buscar", "un momento").
    # Drop whole sentences that match — these are structural signatures of the
    # bot malfunctioning, not client-intent vocabulary. Runs on EVERY exit
    # path (loop and forced answers) because every reply passes through here.
    leak = re.compile(
        r"instruccion interna|reformul\w+ mi respuesta|"
        r"me estas pidiendo que (reformul|reescrib|reh[ai]c)|"
        r"(disculpa|perdona|lo siento|te pido disculpas)[^.!?]*"
        r"(busqued|esperando|sistema|proceso|resultado|interno)|"
        r"estaba esperando|esperando (la|el|los|las) (respuesta|resultado|busqued)|"
        r"(dame|espera|deja(me)?) un (momento|segundo|segundito|instante)|"
        r"un (momento|segundo|segundito|instante)(,| )|"
        r"(mientras|ahora mismo) (lo |las |los )?(reviso|busco|miro|consulto|localizo)|"
        r"voy a (buscar|mirar|revisar|consultar|localizar|enfocarme)|"
        r"estoy (buscando|mirando|revisando|consultando)|"
        r"enseguida (te )?(digo|miro|busco|reviso|localizo)")
    parts = re.split(r"(?<=[.!?\n])\s+", text)
    kept = [p for p in parts if not leak.search(_strip_accents_lower(p))]
    cleaned = " ".join(kept).strip()
    if cleaned != text:
        logger.info("[AGENT] Stripped a leaked internal/process reference from the reply.")
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
                    executor.qty_hints[code] = c["cantidad"]   # authoritative (free JSON)
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
    triage_block = _triage_block(triage)
    if reconciled_msg:
        triage_block = reconciled_msg + triage_block
    # Media turns carry a per-turn DESCRIPCIÓN of the image/audio; a short note
    # points the model at it. (The humility/no-add-without-confirm rules live in
    # the system prompt; messy-list handling is enforced structurally by the
    # batch-quality gate, not by prompt enumeration.)
    if _is_media_message(current_message):
        triage_block = (
            "### LLEGÓ COMO IMAGEN/AUDIO (interpretación, puedes fallar)\n"
            "- Al inicio hay una DESCRIPCIÓN de qué es y lo legible que es. Si es difícil de "
            "leer, díselo con naturalidad y pide una foto más clara o los códigos de producto.\n"
            "- Di qué entendiste y qué no; invita a corregir; nunca apuntes sin confirmación.\n"
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
    # Reserve room for what the LOOP will append on top of this base prompt
    # (search-result digests + tool-call announcements + nudges), so the total
    # context stays under budget once the loop grows it — history is trimmed
    # tail-first now instead of tool results being dropped later.
    _LOOP_RESERVE_TOKENS = _est_tokens("x" * MAX_RESULTS_CHARS) + 200
    avail_tokens = (CTX_BUDGET_TOKENS - _est_tokens(system) - _est_tokens(shell)
                    - _GENERATION_SLACK_TOKENS - _LOOP_RESERVE_TOKENS)
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
    }

    def _fit_and_track() -> None:
        """Trim the accumulated context under budget and record the real peak
        usage (the gauge the console monitors — must stay < CTX_BUDGET_TOKENS,
        i.e. under 70% of the window)."""
        nonlocal messages
        messages = _fit_messages(messages, CTX_BUDGET_TOKENS)
        ctx_stats["used"] = max(int(ctx_stats["used"]), _messages_tokens(messages))

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
        _fit_and_track()
        try:
            resp = await asyncio.to_thread(
                llm.chat, CIMA_MODEL, messages, tools=_TOOLS,
                options={"temperature": AGENT_TEMPERATURE, "top_p": 0.9, "repeat_penalty": 1.1,
                         "num_predict": 512, "stop": ["<NO_REPLY>"]},
            )
        except Exception as e:
            logger.error(f"[AGENT] LLM call failed at step {step}: {e}")
            return await _finish("")

        # The 2B model sometimes writes a tool call as TEXT instead of using the
        # tool channel (it leaked a note-as-text live, and the same failure
        # silently drops search_products/add_item, stalling the order). Recover
        # any tool calls found in the text and merge them so they execute.
        text_calls = _recover_text_tool_calls(resp.content)
        all_calls = (resp.tool_calls or []) + text_calls
        reply = _clean(resp.content)
        queries = executor.apply(all_calls)
        if text_calls:
            logger.info(f"[AGENT] Recovered {len(text_calls)} tool call(s) emitted as text.")
        logger.info(f"[AGENT] step={step} tools={len(all_calls)} "
                    f"queries={queries} status={executor.status} cart={len(executor.cart)} "
                    f"handoff={executor.handoff} opt_out={executor.opt_out}")

        # Trust the model's own handoff_to_human / opt_out_client decisions
        # (guided by the brief prompt) — no triage-flag second-guessing. The
        # only intervention is a safety net: if the model tries to hand off on a
        # turn where it ALSO did real order work, don't strand the order —
        # clear the handoff and let it finish the order in words.
        can_nudge = step < budget - 1
        if executor.handoff and not executor.opt_out and can_nudge and not demoted_mixed \
                and (queries or executor.order_activity or executor.blocked_media_adds
                     or executor.blocked_media_close):
            executor.handoff = False
            demoted_mixed = True
            blocked = ", ".join(dict.fromkeys(executor.blocked_media_adds)) or "lo que pediste"
            executor.blocked_media_adds.clear()
            logger.info("[AGENT] Handoff on an order turn: keeping the order alive.")
            messages.append(Message(role="assistant", content=reply or "(handoff)"))
            messages.append(Message(role="user", content=_internal(
                f"No abandones el pedido. Sigue atendiéndolo con normalidad; si hay algo que solo "
                f"puede resolver {salesman_name} (precio, envío…), dilo en una frase y continúa "
                f"con el pedido — propón lo que entendiste ({blocked}) y espera el 'sí' del "
                "cliente antes de apuntar nada.")))
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
            _fit_and_track()
            try:
                resp = await asyncio.to_thread(
                    llm.chat, CIMA_MODEL, messages,
                    options={"temperature": AGENT_TEMPERATURE, "top_p": 0.9,
                             "repeat_penalty": 1.1, "num_predict": 400})
                forced = _clean(resp.content)
            except Exception as e:
                logger.error(f"[AGENT] Forced capability reply failed: {e}")
                forced = ""
            if not forced:
                # Never leave a redirect turn empty: a plain, leak-free line.
                forced = (f"Puedo localizarte productos por su nombre completo o su código de "
                          f"producto y montarte el pedido aquí mismo. Para precios, stock o "
                          f"incidencias, eso te lo confirma {salesman_name}. ¿Qué producto buscas?")
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
            # Last dispatched order's codes are real and grounded, so allow the
            # client to repeat them without a re-search regardless of the flag.
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
            # Semantic MATCH pass (one simple 0.0 call): decide which retrieved
            # candidates truly are each requested product, instead of guessing
            # from relevance-score thresholds. Uses the merged (per-item) hits
            # and the top few candidates each.
            merged_for_match = _merge_variant_queries(results)
            match_items = [
                {"pedido": q,
                 "candidatos": [_hit_line(h) for h in hits[:6]]}
                for q, hits in merged_for_match.items() if hits
            ]
            match_map = await run_match(match_items) if match_items else None
            # Feed a TRIAGED digest, not a raw hit dump: resolved items in one
            # line each, ONE question at a time, the rest queued — how a real
            # salesman walks a client's list. The queue persists in the session.
            digest, executor.open_items = _triage_results(
                results, executor.open_items,
                is_media=_is_media_message(current_message),
                held_count=len(executor.held_for_enrichment),
                match_map=match_map)
            # Record authoritative quantities (from the free-JSON triage) for
            # every code the search resolved this turn, so add_item uses the
            # real number even if the tool-call grammar truncates its argument.
            for key in ("identificados_seguros", "elegidos_probables"):
                for r in digest.get(key, []):
                    code = _sane_code(r.get("codigo"))
                    if code:
                        executor.qty_hints[code] = _triage_qty(r.get("pedido", ""), triage)
            many_unresolved = "no_encontrados" in digest
            if many_unresolved:
                # >2 unresolved: we already collapsed them into one ask for
                # names/codes. Don't also spell out per-item enrichment prompts
                # (that's the tedious wall we're avoiding) — fold the held names
                # into the same ask and clear them.
                if executor.held_for_enrichment:
                    digest["no_encontrados"] = list(digest["no_encontrados"]) + \
                        list(executor.held_for_enrichment)
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

        if not reply and all_calls and step < MAX_STEPS - 1:
            # Small models often emit only tool calls first; ask for the final text.
            messages.append(Message(role="assistant", content=json.dumps(
                {"tool_calls": [c.get("function", {}).get("name") for c in all_calls]})))
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
            "y si pidió un resumen, dáselo a partir del carrito actual. PROHIBIDO prometer que "
            "vas a buscar, pedir que espere ('un momento', 'dame un segundo'), disculparte por "
            "demoras o mencionar búsquedas, sistemas o procesos internos: la búsqueda ya ha "
            "terminado, responde con lo que TIENES. NO llames a ninguna herramienta; responde "
            "solo con el texto para el cliente.")))
        _fit_and_track()
        try:
            resp = await asyncio.to_thread(
                llm.chat, CIMA_MODEL, messages,   # NO tools: text is the only valid output
                options={"temperature": AGENT_TEMPERATURE, "top_p": 0.9,
                         "repeat_penalty": 1.1, "num_predict": 512},
            )
            reply = _clean(resp.content)
        except Exception as e:
            logger.error(f"[AGENT] Forced final-answer call failed: {e}")
        if len(reply.strip()) < 20:
            # Empty, or cleaned down to a stub like "Entendido." after a leak
            # was stripped — a deterministic state summary is more useful.
            greet = intro_mode == "new" or (
                intro_mode == "renew" and (executor.order_activity or executor.results_this_turn))
            reply = _fallback_reply(list(executor.cart.values()), executor.open_items,
                                    client_name, salesman_name, greet)
            logger.warning("[AGENT] Forced reply empty/stub; using deterministic fallback.")

    reply, leaked_note = _split_leaked_note(reply)
    if leaked_note and not executor.summary_touched:
        executor.summary = leaked_note[:300]
        logger.info(f"[AGENT] Leaked visible note rerouted to summary: '{leaked_note[:60]}'")
    return await _finish(reply)