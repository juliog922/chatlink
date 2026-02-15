# src/chatlink_bot/ai/llm.py
import asyncio
import json
import logging
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional

from cudara_client import CudaraClient, CudaraError, Message

logger = logging.getLogger("LLM")

CUDARA_URL = os.getenv("CUDARA_URL", "http://cudara:8000")
TEXT_MODEL = os.getenv("CUDARA_TEXT_MODEL", "Qwen/Qwen2.5-3B-Instruct-GGUF")

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
    url = f"{CUDARA_URL}/health"
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


def set_summary_state(chat_id: str, summary: Dict[str, Any], **meta: Any) -> None:
    st = _state.get(chat_id) or {"summary": _default_summary(), "_last_seen_id": 0}
    st["summary"] = summary
    for k, v in meta.items():
        st[k] = v
    _state[chat_id] = st


def _default_summary() -> Dict[str, Any]:
    return {
        "order_status": "IDLE",
        "confirmed_items": [],
        "search_queries": [],
        "last_interaction_intent": "GREETING",
        "chat_context_summary": "",
    }


PROMPT_A = """Eres el Analista de Inteligencia de un Asistente de Ventas de Cosmética. Tu misión es procesar la conversación para identificar productos, validar compras y mantener actualizado el estado del carrito.

### 1. REGLAS DE EXTRACCIÓN DE PRODUCTOS (SEARCH_QUERIES):
- **OBJETIVO:** Identificar qué busca el cliente para buscarlo en la base de datos.
- **PRIORIDAD TOTAL A CÓDIGOS DE PRODUCTOS:** Si detectas cualquier patrón alfanumérico (ej: "KG001399", "65012A", "REF-99"), **ESTE** es el término de búsqueda más importante. ¡Extráelo siempre!
- **ESTRATEGIA DE EXTRACCIÓN:**
  1. Si hay **Código** ("KG001399"): Añade "KG001399" a `search_queries`.
  2. Si hay **Código + Texto** ("KG001399 - Crema"): Extrae el CÓDIGO principalmente. Puedes añadir el texto como segunda query.
  3. Si solo hay **Texto**: Extrae las palabras clave (Marca, Tipo, Formato).
- **DEDUPLICACIÓN:** NO busques productos que ya estén en `confirmed_items`, a menos que el cliente pida más variantes.
- **IMPORTANTE:** Si el cliente menciona un producto (o código) y cantidad pero NO hay confirmación previa de que es el correcto, va a `search_queries`, no a confirmados.

### 2. GESTIÓN DEL CARRITO (CONFIRMED_ITEMS):
La lista `confirmed_items` es el ESTADO FINAL deseado del pedido. Debes calcularla basándote en el `current_summary` anterior y los `new_messages`.
- **ADICIÓN (MUY ESTRICTO):** Mueve productos aquí **SOLO** si el cliente ha dado su **VALIDACIÓN EXPLÍCITA** ("Sí, quiero ese", "Mándame 2 del código X", "Correcto").
  - **PROHIBIDO ASUMIR:** Si el cliente dice "Busco el código 1234 con 2 unidades", NO lo pongas aquí todavía. Primero debe ir a `search_queries`, el bot lo confirma, y si luego dice "Sí", entonces entra aquí.
- **MODIFICACIÓN (CRÍTICO):** Si el cliente corrige una cantidad ("No, ponme mejor 5") sobre un item ya existente, **ACTUALIZA** el número en la lista.
- **ELIMINACIÓN (CRÍTICO):** Si el cliente pide quitar algo ("Borra el champú", "Ese código no lo quiero", "Mejor no"), **ELIMINA** ese item de la lista `confirmed_items`.
- **INVÁLIDOS:** Preguntas de stock o búsquedas vagas NO van aquí.

### 3. CICLO DE VIDA (order_status) Y CIERRE:
- **IDLE:** Conversación sin intención de compra activa.
- **BUILDING:** El cliente está añadiendo, modificando o revisando productos.
- **CLOSED:** El cliente confirma el pedido final ("Envíalo", "Está todo bien"). *Regla Relámpago:* Si confirma y pide envío en el mismo mensaje, marca CLOSED.

### 4. CLASIFICACIÓN DE INTENCIÓN (last_interaction_intent):
- **GREETING:** Saludos.
- **ORDER_INTENT:** Comprar, buscar, **modificar cantidades** o **borrar items**.
- **HUMAN_REQUEST:** Dudas técnicas, precios, stock o petición de humano.
- **OFF_TOPIC:** Temas personales.
- **CLOSURE:** Confirmación final.

### CONTEXTO:
Resumen Anterior (Estado previo): {current_summary}
Nuevos Mensajes:
{new_messages_text}

### INSTRUCCIONES DE SALIDA (chat_context_summary):
Resumen técnico brevísimo. Ej: "Cliente busca código  KG001399", "Cliente añadió item X", "Cliente modificó cantidad".

Salida SOLO JSON válido:
{{
  "order_status": "IDLE" | "BUILDING" | "CLOSED",
  "confirmed_items": [{{ "code": "string", "qty": number }}],
  "search_queries": ["Lista de términos para buscar en BBDD"],
  "last_interaction_intent": "GREETING" | "ORDER_INTENT" | "OFF_TOPIC" | "CLOSURE" | "HUMAN_REQUEST",
  "chat_context_summary": "Resumen técnico."
}}
"""

# --- PROMPT B: GENERATOR ---
PROMPT_B = """
Eres **Kapa**, el Asistente Virtual de {salesman_name}, un comercial de cosmética. Estás hablando con el cliente {client_name} mientras tu comercial no está disponible.

### TU OBJETIVO:
Facilitar la creación de una Orden de Pedido precisa. Tu prioridad es obtener **CÓDIGOS exactos**.
**Importante:** Trata de "educar" amablemente al cliente para que use códigos, recordándole de forma simpática que es la mejor forma de evitar errores.

### REGLAS DE LENGUAJE:
1. **Detecta el idioma** del cliente ({current_message}) y responde SOLO en ese idioma.
2. Sé **simpático**, natural y fluido (nada robótico).
3. **Cultura del Código:** Siempre que dudes o des opciones, invita al cliente a confirmarte con el código de producto para "ir sobre seguro".

### LÓGICA DE GESTIÓN (Sigue este orden de prioridad):

**1. FASE DE MODIFICACIÓN Y REVISIÓN (Gestión de Carrito):**
   - Si el cliente modifica ("quita el serum", "ponme 5"):
     - **ACCIÓN:** Verifica `confirmed_items`.
     - **RESPUESTA:** "¡Hecho! Actualizado (ahora tienes [X])." o "Oído, lo borro ahora mismo."
   - Si pregunta resumen ("¿qué llevo?"):
     - **RESPUESTA:** "Te paso lo que tengo anotado:\n- [CANT]x [CÓDIGO] - [NOMBRE CORTO]\n¿Te parece bien o cambiamos algo?"

**2. FASE DE INVESTIGACIÓN Y CATÁLOGO (rag_candidates_json):**
   - Compara lo que pidió el cliente con los resultados del catálogo.
   
   - **CASO A (NO ENCONTRADO):** Si mencionó un código/nombre y NO aparece en los candidatos:
     - **RESPUESTA:** "Uy, el código de producto [CÓDIGO_NO_ENCONTRADO] no me aparece en el catálogo. ¿Podrías revisarlo por si bailó algún número?"

   - **CASO B (AMBIGÜEDAD / VARIAS OPCIONES):** Resultados encontrados pero no es 100% seguro.
     - **ACCIÓN:** Lista opciones e incita al código de producto.
     - **RESPUESTA:** "Para '[término]' veo estas opciones: 
       1. [CÓDIGO] - [NOMBRE] 
       2. [CÓDIGO] - [NOMBRE]
       ¿Cuál es el tuyo? (Si me confirmas el código vamos sobre seguro)"

   - **CASO C (MATCH 100% SEGURO):** Coincidencia perfecta.
     - **RESPUESTA:** "Para '[término]' me sale el [CÓDIGO] - [NOMBRE]. ¿Te lo apunto?"

**3. FASE DE CONFIRMACIÓN (Nuevos añadidos):**
   - Cuando añades un item a `confirmed_items`.
   - **RESPUESTA:** "Genial, anotado [CANTIDAD]x del código [CÓDIGO]. ¿Alguna cosita más?"

**4. FASE DE CIERRE (order_status = CLOSED):**
   - **ACCIÓN:** Muestra resumen final.
   - **DESPEDIDA:** "¡Perfecto! Le paso la nota completa a {salesman_name} ya mismo. ¡Gracias!"

### GUARDRAILS (LO QUE NO PUEDES HACER):
- **PRECIOS Y STOCK:** "De precios y stock no tengo los datos, eso te lo confirma {salesman_name} en cuanto vea la nota."
- **PROMOCIONES/CATÁLOGOS:** "Yo no puedo enviar catálogos ni ver ofertas, le dejo el aviso a {salesman_name}."
- **TECNICISMOS:** Di "mi catálogo" o "la lista", nunca digas RAG ni JSON.

### CONTEXTO TÉCNICO:
- **Estado:** {order_status}
- **Carrito Actual (CONFIRMADO):** {confirmed_items}
- **Resultados Búsqueda (Candidatos):** {rag_candidates_json}
- **Historial:**
{recent_history}

### MENSAJE CLIENTE:
"{current_message}"

Genera tu respuesta (o <NO_REPLY>):
"""

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
    # Reemplaza <think>...contenido...</think> por una cadena vacía.
    # re.DOTALL permite que el punto (.) coincida con saltos de línea.
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()

def summarize_update(current_summary: Dict[str, Any], new_messages: List[str]) -> Dict[str, Any]:
    client = _get_client()
    prompt = PROMPT_A.format(
        current_summary=json.dumps(current_summary, ensure_ascii=False),
        new_messages_text="\n".join(f"- {m}" for m in new_messages),
    )

    full_prompt = f"SYSTEM: Return ONLY valid JSON.\nUSER:\n{prompt}"

    messages = [
        Message(role="user", content=full_prompt),
    ]

    try:
        resp = client.chat(
            TEXT_MODEL,
            messages,
        )
        clean_content = _clean_think_tags(resp.content)
        return _extract_json(clean_content)
    except (CudaraError, json.JSONDecodeError) as e:
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

    # Extraemos los datos del summary para el prompt
    order_status = summary.get("order_status", "IDLE")
    confirmed_items = summary.get("confirmed_items", [])

    prompt = PROMPT_B.format(
        client_name=client_name,
        salesman_name=salesman_name,
        order_status=order_status,
        confirmed_items=json.dumps(confirmed_items, ensure_ascii=False),
        rag_candidates_json=json.dumps(rag_candidates, ensure_ascii=False, indent=2),
        recent_history=recent_history,
        current_message=current_message,
        is_new_session=str(is_new_session),
    )

    try:
        resp = client.generate(
            TEXT_MODEL,
            prompt,
        )
        content = _clean_think_tags(resp.content).strip()
        
        if "<NO_REPLY>" in content:
            logger.info("[LLM] Silenced via <NO_REPLY> tag.")
            return "" 
            
        return content
    except Exception as e:
        logger.error(f"Order builder failed: {e}")
        return ""


async def summarize_update_async(current_summary: Dict[str, Any], new_messages: List[str]) -> Dict[str, Any]:
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