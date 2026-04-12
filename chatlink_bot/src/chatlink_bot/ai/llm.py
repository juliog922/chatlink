# src/chatlink_bot/ai/llm.py
import asyncio
import json
import logging
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional

from cudara_client import CudaraClient, Message

logger = logging.getLogger("LLM")

CUDARA_URL = os.getenv("CUDARA_URL", "http://cudara:8000")
TEXT_MODEL = os.getenv("CUDARA_TEXT_MODEL", "Qwen/Qwen2.5-3B-Instruct-GGUF")
# Default to 4096, matching the llama.cpp standard and your error logs
CTX_WINDOW = int(os.getenv("CUDARA_CTX_WINDOW", "4096"))

_client: Optional[CudaraClient] = None

# chat_id -> {"summary": {...}, "_last_seen_id": int}
_state: Dict[str, Dict[str, Any]] = {}


def _get_client() -> CudaraClient:
    global _client
    if _client is None:
        _client = CudaraClient(CUDARA_URL)
    return _client


async def health_check() -> Dict[str, Any]:
    """
    Checks if Cudara service is reachable via its /health endpoint.
    """
    url = f"{CUDARA_URL}/api/version"  # Updated endpoint
    try:
        def _fetch():
            with urllib.request.urlopen(url, timeout=3) as resp:
                return resp.getcode()

        code = await asyncio.to_thread(_fetch)
        if code == 200:
            return {"status": "ok", "code": code, "url": url}
        return {"status": "error", "code": code, "url": url}
    except Exception as e:
        return {"status": "error", "error": str(e), "url": url}


def get_summary_state(chat_id: str) -> Dict[str, Any]:
    return _state.get(chat_id) or {"summary": _default_summary(), "_last_seen_id": 0}


def _prune_state():
    """Prevents memory leaks by keeping max 1000 active summaries."""
    if len(_state) > 1000:
        # Delete the 100 oldest, most inactive conversations
        keys_to_delete = list(_state.keys())[:100]
        for k in keys_to_delete:
            _state.pop(k, None)


def set_summary_state(chat_id: str, summary: Dict[str, Any], **meta: Any) -> None:
    _prune_state()

    # If it exists, pop it so we can re-insert it at the end (marking it as recently used)
    st = _state.pop(chat_id, {"summary": _default_summary(), "_last_seen_id": 0})

    st["summary"] = summary
    for k, v in meta.items():
        st[k] = v

    # Re-insert at the end of the dictionary
    _state[chat_id] = st


def _default_summary() -> Dict[str, Any]:
    return {
        "order_status": "IDLE",
        "confirmed_items": [],
        "search_queries": [],
        "last_interaction_intent": "GREETING",
        "chat_context_summary": "",
    }


def _estimate_tokens(text: str) -> int:
    """Quick and dirty token estimation (approx 4 chars per token)."""
    return len(str(text)) // 4


def _truncate_rag_candidates(candidates: Dict[str, Any], token_budget: int) -> Dict[str, Any]:
    """Reduces the number of RAG candidates to fit the token budget."""
    truncated = {}
    current_tokens = 0

    for query, items in candidates.items():
        block_str = json.dumps({query: items}, ensure_ascii=False)
        block_tokens = _estimate_tokens(block_str)

        if current_tokens + block_tokens < token_budget:
            truncated[query] = items
            current_tokens += block_tokens
        else:
            # If the full block doesn't fit, try taking just the top 1 item for this query
            if items:
                small_block = json.dumps({query: items[:1]}, ensure_ascii=False)
                small_tokens = _estimate_tokens(small_block)
                if current_tokens + small_tokens < token_budget:
                    truncated[query] = items[:1]
                    current_tokens += small_tokens
            break  # Stop adding queries once budget is hit

    return truncated


# ===================================================================
# PROMPT A — Conversation Analyzer / Summarizer
# ===================================================================
# Separated into SYSTEM and USER for proper chat formatting.

_PROMPT_A_SYSTEM = """\
Eres el Motor de Análisis de un Asistente de Ventas de Cosmética profesional. \
Tu única función es procesar conversaciones entre clientes y comerciales para \
producir un JSON estructurado que mantiene el estado del pedido.

REGLAS DE SALIDA:
- Responde EXCLUSIVAMENTE con un objeto JSON válido. Sin texto adicional, sin markdown, sin explicaciones.
- Nunca inventes productos, códigos ni cantidades que no aparezcan en los mensajes."""

_PROMPT_A_USER = """\
### 1. EXTRACCIÓN DE PRODUCTOS (search_queries)
OBJETIVO: Identificar qué busca el cliente para poder consultar el catálogo de cosmética.

PRIORIDAD ABSOLUTA → CÓDIGOS DE PRODUCTO:
- Cualquier patrón alfanumérico que parezca referencia de producto (ej: "KG001399", "65012A", "REF-99", "LC320", "BB-045") va SIEMPRE como primer elemento en `search_queries`.
- Estrategia de extracción:
  • Código solo ("KG001399") → search_queries: ["KG001399"]
  • Código + descripción ("KG001399 - Crema hidratante") → search_queries: ["KG001399", "Crema hidratante"]
  • Solo texto descriptivo ("crema antiedad Kérastase") → search_queries: ["crema antiedad Kérastase"]
  • Múltiples productos en un mensaje → extrae CADA uno como query separada.

MULTIMEDIA: Los bloques marcados con [Texto en Imagen], [Audio transcrito] o [Documento] contienen \
información extraída de fotos, audios o archivos que el cliente envió. Trátalo EXACTAMENTE igual que \
texto escrito por el cliente. Es MUY común que los clientes envíen fotos de listas de pedidos, \
catálogos o facturas anteriores — extrae TODOS los códigos y productos que aparezcan.

DEDUPLICACIÓN: NO busques productos que ya estén en `confirmed_items` salvo que el cliente pida variantes.

### 2. GESTIÓN DEL CARRITO (confirmed_items)
La lista `confirmed_items` es el ESTADO DEFINITIVO del pedido. Cada item tiene: {{"code": "string", "qty": number}}

REGLAS ESTRICTAS:
- ADICIÓN: Solo cuando el cliente da VALIDACIÓN EXPLÍCITA ("sí, ese", "mándame 2", "correcto, apúntalo").
  → PROHIBIDO ASUMIR: "Busco el código 1234 con 2 unidades" NO se confirma automáticamente. \
    Primero va a search_queries, el bot presenta el resultado, y solo con confirmación explícita entra aquí.
- MODIFICACIÓN: Si el cliente corrige cantidad ("ponme 5 en vez de 3"), ACTUALIZA el qty del item existente.
- ELIMINACIÓN: Si el cliente quita algo ("borra el champú", "ese no"), ELIMINA el item de la lista.
- CONSERVACIÓN: Items confirmados anteriormente se MANTIENEN intactos salvo modificación o eliminación explícita.

### 3. CICLO DE VIDA (order_status)
- "IDLE": Sin intención de compra activa.
- "BUILDING": El cliente busca, añade, modifica o revisa productos. Cualquier actividad de pedido.
- "CLOSED": El cliente confirma EXPLÍCITAMENTE el pedido final ("Envíalo", "Todo correcto, ciérralo"). \
  Si confirma y pide envío en el mismo mensaje → CLOSED directo.

### 4. CLASIFICACIÓN DE INTENCIÓN (last_interaction_intent)
Clasifica el ÚLTIMO mensaje del cliente:
- "GREETING": EXCLUSIVAMENTE saludos puros o despedidas sin petición adicional ("Hola", "Buenos días", "Gracias").
- "ORDER_INTENT": CUALQUIER mención a productos, catálogo, búsquedas, cantidades, modificaciones de carrito, \
  o peticiones de resumen de pedido. Incluye fotos/audios con listas de productos.
- "HUMAN_REQUEST": El cliente se dirige EXPLÍCITAMENTE al comercial por nombre, pide hablar con una persona, \
  pregunta por facturación/pagos, o plantea cuestiones que exceden la gestión del carrito.
- "OFF_TOPIC": Temas completamente ajenos al negocio (vida personal, política, deportes).
- "CLOSURE": Confirmación final EXCLUSIVA para cerrar y enviar el pedido completo.

### 5. RESUMEN TÉCNICO (chat_context_summary)
Frase brevísima describiendo la acción: "Cliente busca código KG001399", "Cliente confirmó 3 items", \
"Cliente eliminó champú del carrito".

---

ESTADO ANTERIOR:
{current_summary}

NUEVOS MENSAJES:
{new_messages_text}

Responde SOLO con el JSON:
{{"order_status": "...", "confirmed_items": [...], "search_queries": [...], "last_interaction_intent": "...", "chat_context_summary": "..."}}"""


# ===================================================================
# PROMPT B — Reply Generator (Kapa, the sales assistant)
# ===================================================================

_PROMPT_B_SYSTEM = """\
Eres **Kapa**, el Asistente Virtual de un comercial de cosmética. Tu misión principal es ayudar \
a los clientes a construir su pedido de productos de forma rápida, precisa y con buen rollo.

PERSONALIDAD:
- Simpática, eficiente, profesional pero cercana (como una compañera de trabajo que echa un cable).
- Usa un tono natural de WhatsApp: frases cortas, algún emoji puntual (sin abusar), lenguaje fluido.
- NUNCA robótica ni corporativa. NUNCA menciones términos técnicos (RAG, JSON, base de datos, sistema).

REGLAS FUNDAMENTALES:
1. DETECTA EL IDIOMA del mensaje del cliente y responde SOLO en ese idioma.
2. CÓDIGOS son tu mejor amigo: siempre invita al cliente a confirmar con el código de producto \
   para evitar confusiones. Hazlo de forma natural, no machacona.
3. CONTINUIDAD: Si la conversación ya está en marcha (Es sesión nueva: False), NO te presentes \
   de nuevo. Continúa directamente desde donde se quedó.
4. MULTIMEDIA: Si el mensaje incluye [Texto en Imagen] o [Audio transcrito], demuestra que \
   lo has procesado ("¡Visto! De esa foto saco estos códigos...", "Perfecto, he escuchado tu audio").
5. LONGITUD: Sé concisa. En WhatsApp nadie quiere párrafos. Respuestas directas y al grano.

GUARDRAILS (LO QUE NO PUEDES HACER):
- PRECIOS Y STOCK: "Eso te lo confirma {salesman_name} cuando vea la nota, yo no tengo esos datos."
- PROMOCIONES/CATÁLOGOS: "No puedo enviarte catálogos ni ver ofertas, le aviso a {salesman_name}."
- ASESORAMIENTO TÉCNICO: Si preguntan qué producto les conviene más, sugiere consultar con el comercial.
- NUNCA inventes información sobre productos que no aparezca en los Resultados de Búsqueda.

SILENCIO (<NO_REPLY>): Responde EXACTAMENTE "<NO_REPLY>" (sin nada más) cuando:
- El mensaje es claramente OFF_TOPIC y no merece respuesta.
- El comercial acaba de intervenir y no hay nada útil que añadir."""

_PROMPT_B_USER = """\
### PROTOCOLO DE RESPUESTA (sigue este orden de prioridad):

**0. SALUDO (si el Estado es IDLE y Es sesión nueva: True):**
   - Preséntate brevemente: "¡Hola {client_name}! Soy Kapa, el asistente de {salesman_name}. Cuéntame, ¿qué necesitas para tu pedido?"
   - Si Es sesión nueva: False → saludo de continuación natural sin presentarte.

**1. GESTIÓN DE CARRITO (si el cliente modifica, quita o pregunta por su pedido):**
   - Modificación: "¡Hecho! Te actualizo: ahora llevas [X] unidades del [CÓDIGO]."
   - Eliminación: "Oído, lo quito del pedido."
   - Resumen: Lista el carrito limpio:
     "Esto es lo que llevas:
     • [CANT]x [CÓDIGO] — [NOMBRE]
     • [CANT]x [CÓDIGO] — [NOMBRE]
     ¿Añadimos algo más o cerramos?"

**2. BÚSQUEDA EN CATÁLOGO (usa los Resultados de Búsqueda):**
   Compara lo que el cliente pidió con los resultados del catálogo.

   - **NO ENCONTRADO** (0 resultados o nada relevante):
     "El código [X] no me aparece en el catálogo. ¿Podrías revisarlo por si bailó algún número?"

   - **VARIAS OPCIONES** (resultados similares pero sin match perfecto):
     "Para '[término]' veo estas opciones:
     1. [CÓDIGO] — [NOMBRE] ([MARCA])
     2. [CÓDIGO] — [NOMBRE] ([MARCA])
     ¿Cuál es? Si me dices el código vamos sobre seguro."

   - **MATCH EXACTO** (1 resultado claro):
     "Para '[término]' me sale el [CÓDIGO] — [NOMBRE]. ¿Te lo apunto? ¿Cuántas unidades?"

**3. CONFIRMACIÓN DE NUEVOS ITEMS:**
   - Cuando el cliente confirma: "¡Anotado! [CANT]x [CÓDIGO]. ¿Algo más?"

**4. CIERRE (Estado = CLOSED):**
   - Resumen final completo + despedida:
     "¡Perfecto! Tu pedido queda así:
     • [CANT]x [CÓDIGO] — [NOMBRE]
     Le paso la nota a {salesman_name} ahora mismo. ¡Gracias {client_name}!"

---

### CONTEXTO:
- **Estado pedido:** {order_status}
- **Es sesión nueva:** {is_new_session}
- **Resumen conversación:** {chat_context_summary}
- **Carrito confirmado:** {confirmed_items}
- **Resultados de Búsqueda:** {rag_candidates_json}
- **Historial reciente:**
{recent_history}

### MENSAJE DEL CLIENTE:
"{current_message}"

Tu respuesta (o <NO_REPLY>):"""


def _extract_json(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def _clean_think_tags(text: str) -> str:
    """
    Elimina el contenido entre los tags <think> y </think>, incluyendo los propios tags.
    Útil para modelos de razonamiento que exponen su cadena de pensamiento.
    """
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def summarize_update(current_summary: Dict[str, Any], new_messages: List[str]) -> Dict[str, Any]:
    client = _get_client()

    # Pre-flight budget check for prompt A (Summarizer)
    target_budget = int(CTX_WINDOW * 0.7)

    # Truncate new_messages if someone pasted a massive block of text
    new_msgs_text = "\n".join(f"- {m}" for m in new_messages)
    if _estimate_tokens(new_msgs_text) > (target_budget * 0.4):
        logger.warning("Summarizer: Truncating massively long new messages to fit context window.")
        new_msgs_text = new_msgs_text[: int(target_budget * 0.4 * 4)] + "... [TRUNCATED]"

    user_content = _PROMPT_A_USER.format(
        current_summary=json.dumps(current_summary, ensure_ascii=False),
        new_messages_text=new_msgs_text,
    )

    messages = [
        Message(role="system", content=_PROMPT_A_SYSTEM),
        Message(role="user", content=user_content),
    ]

    try:
        resp = client.chat(
            TEXT_MODEL,
            messages,
            format="json",
            options={
                "temperature": 0.0,      # Deterministic — faithful extraction
                "top_p": 1.0,
                "repeat_penalty": 1.0,    # Don't penalize repeated codes/items
                "num_predict": 1024,      # JSON summaries don't need to be long
            },
        )
        clean_content = _clean_think_tags(resp.content)
        parsed = _extract_json(clean_content)

        # -----------------------------------------------------------
        # Validate and sanitize every field from the LLM output
        # -----------------------------------------------------------
        validated = _default_summary()

        validated["order_status"] = str(parsed.get("order_status", validated["order_status"]))
        validated["last_interaction_intent"] = str(
            parsed.get("last_interaction_intent", validated["last_interaction_intent"])
        )
        validated["chat_context_summary"] = str(
            parsed.get("chat_context_summary", validated["chat_context_summary"])
        )

        # Validate confirmed_items structure
        raw_items = parsed.get("confirmed_items", [])
        if isinstance(raw_items, list):
            clean_items = []
            for item in raw_items:
                if isinstance(item, dict) and "code" in item:
                    clean_items.append({
                        "code": str(item["code"]),
                        "qty": int(item.get("qty", 1)),
                    })
            validated["confirmed_items"] = clean_items
        else:
            # LLM returned garbage — preserve what we had before
            validated["confirmed_items"] = current_summary.get("confirmed_items", [])

        # Validate search_queries
        raw_queries = parsed.get("search_queries", [])
        if isinstance(raw_queries, list):
            validated["search_queries"] = [str(q) for q in raw_queries if q and str(q).strip()]
        else:
            validated["search_queries"] = []

        # Validate order_status is a known value
        if validated["order_status"] not in ("IDLE", "BUILDING", "CLOSED"):
            logger.warning(f"Summarizer returned unknown order_status: {validated['order_status']}")
            validated["order_status"] = current_summary.get("order_status", "IDLE")

        # Validate intent is a known value
        valid_intents = {"GREETING", "ORDER_INTENT", "OFF_TOPIC", "CLOSURE", "HUMAN_REQUEST"}
        if validated["last_interaction_intent"] not in valid_intents:
            logger.warning(
                f"Summarizer returned unknown intent: {validated['last_interaction_intent']}"
            )
            validated["last_interaction_intent"] = "ORDER_INTENT"  # Safe default

        return validated

    except (Exception, json.JSONDecodeError) as e:
        logger.error(f"Summarizer failed: {e}")
        return current_summary


def build_order_reply(
    client_name: str,
    salesman_name: str,
    summary: Dict[str, Any],
    rag_candidates: Dict[str, Any],
    recent_history: str,
    current_message: str,
    is_new_session: bool,
) -> str:

    logger.info(f"[PROMPT DEBUG] Focus Msg: '{current_message}' | Status: {summary.get('order_status')}")

    client = _get_client()

    # Budget Target: 70% of available context
    target_budget = int(CTX_WINDOW * 0.7)

    order_status = summary.get("order_status", "IDLE")
    confirmed_items = summary.get("confirmed_items", [])
    chat_context_summary = summary.get("chat_context_summary", "")

    # Split history into distinct lines/messages
    history_lines = [line for line in recent_history.split("\n") if line.strip()]

    # System prompt with salesman name baked in
    system_content = _PROMPT_B_SYSTEM.format(salesman_name=salesman_name)

    # Helper function to dynamically build the user prompt string for token estimation
    def _build_user_content(h_lines: List[str], r_cands: Dict[str, Any]) -> str:
        r_json = json.dumps(r_cands, ensure_ascii=False, indent=2) if r_cands else "Sin resultados."
        h_text = "\n".join(h_lines) if h_lines else "(Sin historial previo)"
        return _PROMPT_B_USER.format(
            client_name=client_name,
            salesman_name=salesman_name,
            order_status=order_status,
            confirmed_items=json.dumps(confirmed_items, ensure_ascii=False) if confirmed_items else "[]",
            rag_candidates_json=r_json,
            recent_history=h_text,
            current_message=current_message,
            is_new_session=str(is_new_session),
            chat_context_summary=chat_context_summary or "(Inicio de conversación)",
        )

    def _full_prompt_tokens(h_lines: List[str], r_cands: Dict[str, Any]) -> int:
        return _estimate_tokens(system_content) + _estimate_tokens(
            _build_user_content(h_lines, r_cands)
        )

    # 1. Initial Build — check if we need to trim
    if _full_prompt_tokens(history_lines, rag_candidates) > target_budget:
        logger.warning(
            f"Initial prompt too large ({_full_prompt_tokens(history_lines, rag_candidates)} tokens). "
            f"Trimming history first..."
        )

        # STEP A: Trim history down to a minimum of 3 messages (Drops oldest first)
        while (
            len(history_lines) > 3
            and _full_prompt_tokens(history_lines, rag_candidates) > target_budget
        ):
            history_lines.pop(0)

        # STEP B: If STILL too big, trim RAG candidates
        if _full_prompt_tokens(history_lines, rag_candidates) > target_budget:
            logger.warning("Still exceeds budget after history trim. Trimming RAG candidates...")

            tokens_without_rag = _full_prompt_tokens(history_lines, {})
            allowed_rag_tokens = target_budget - tokens_without_rag

            if allowed_rag_tokens > 200:
                rag_candidates = _truncate_rag_candidates(rag_candidates, allowed_rag_tokens)
            else:
                rag_candidates = {}

        # STEP C: Final Emergency safety net
        if _full_prompt_tokens(history_lines, rag_candidates) > target_budget:
            logger.error(f"Prompt STILL exceeds {target_budget} budget. Emergency RAG purge.")
            rag_candidates = {}

    user_content = _build_user_content(history_lines, rag_candidates)

    messages = [
        Message(role="system", content=system_content),
        Message(role="user", content=user_content),
    ]

    try:
        resp = client.chat(
            TEXT_MODEL,
            messages,
            options={
                "temperature": 0.6,       # Warm but not wild — natural WhatsApp tone
                "top_p": 0.9,
                "repeat_penalty": 1.1,     # Light repeat penalty for natural variety
                "num_predict": 512,        # WhatsApp replies should be concise
            },
        )
        content = _clean_think_tags(resp.content).strip()

        if "<NO_REPLY>" in content:
            logger.info("[LLM] Silenced via <NO_REPLY> tag.")
            return ""

        return content
    except Exception as e:
        logger.error(f"Order builder failed: {e}")
        return ""


async def summarize_update_async(
    current_summary: Dict[str, Any], new_messages: List[str]
) -> Dict[str, Any]:
    return await asyncio.to_thread(summarize_update, current_summary, new_messages)


async def build_order_reply_async(
    client_name: str,
    salesman_name: str,
    summary: Dict[str, Any],
    rag_candidates: Dict[str, Any],
    recent_history: str,
    current_message: str,
    is_new_session: bool,
) -> str:
    return await asyncio.to_thread(
        build_order_reply,
        client_name,
        salesman_name,
        summary,
        rag_candidates,
        recent_history,
        current_message,
        is_new_session,
    )