# NOTES — Kapa (bot de pedidos ChatLink)

> Actualizado 2026-08-24. Este documento manda sobre el código. Si el código
> hace algo que este documento no dice, el código está mal. Escrito tras
> comprobar que el diseño anterior (extracción sin conversación + reglas
> estructurales acumuladas) se volvió rígido y antinatural: el bot no leía el
> hilo, así que "sí ese", "13", "quítame el otro" se caían y se intentaban
> tapar con más reglas. La corrección de raíz está en §3 (el loop): **el LLM
> lee la conversación; el código escribe el estado y verifica.**

---

## 1. Qué es Kapa y qué NO es

Kapa cubre al comercial **solo cuando este no ha contestado** por su canal
(WhatsApp o Gmail) en un tiempo configurado. Su única función es **montar
pedidos** y pasárselos al comercial. No es atención al cliente.

Reglas de producto, todas obligatorias:

1. **Solo habla si el comercial no ha respondido** en `RESPONSE_DELAY_MINUTES`.
   Si el comercial contesta antes, Kapa calla (lo gobierna el FSM/debounce, no
   el LLM).
2. **Solo temas de pedido.** Cualquier otra cosa comercial que Kapa no puede
   atender —precios, especificaciones de producto, plazos, facturas, quejas—
   se **deriva** al comercial ("se pondrá en contacto en breve", redactado
   por el LLM, **nunca** una frase fija).
3. **No responde si el cliente pide hablar con el comercial** explícitamente.
   Calla y el comercial toma el hilo.
4. **No responde a charla no comercial.** Si no hay nada de pedido que hacer
   y el comercial está gestionando, Kapa calla. El bot contesta POCO: hablar
   es la excepción, no la norma.
5. **Confirmación antes de cerrar.** Una vez cerrado el pedido no hay
   modificación por parte del bot; para rectificar, se espera al comercial.
   Por eso el cierre siempre pide confirmación explícita.
6. **El comercial es el último filtro.** Todo pedido lo valida él antes de
   salir. Kapa es permisivo al apuntar y honesto cuando duda.

Nada de respuestas hardcodeadas. Las frases fijas se ven falsas y, peor,
hacen que el bot "responda sin leer". El LLM redacta; el código solo verifica
que no invente códigos ni cantidades (§3, etapa REDACTAR/VERIFICAR).

---

## 2. Cómo son los clientes y los datos (por qué esto es difícil)

- Los clientes **ya han hablado con el comercial** antes, una o varias veces.
  Piden con abreviaturas, apodos y nombres que **no** coinciden con el nombre
  de catálogo ("el kajal negro", "brow 403", "lo de siempre").
- Mandan **listas largas** (10-15 productos de golpe) por **texto, imagen o
  documento**. La media y los documentos ya llegan transcritos con marcador.
- La base por cliente son **sus pedidos pasados**, filtrados a lo visible por
  el bot (`K_BOT == 0`). Hoy es lo único fiable (el catálogo general está
  sucio). **Más adelante** se ampliará al catálogo total con el filtro del
  bot; el loop no debe asumir que la fuente es solo el historial, solo que
  "hay un recuperador de productos del cliente".
- El cliente **se refiere a cosas anteriores**: "quítame el otro", "agrégame
  2 más", "como el pedido pasado". El bot **tiene que mantener el hilo**.
- Puede pedir **rehacer un pedido anterior**. El bot los conoce por orden
  cronológico, pero **siempre pide confirmación** antes de rehacerlo.

Consecuencia de diseño: sin la conversación delante, el LLM no puede resolver
nada de esto. Por eso el hilo entra en el prompt (§3).

---

## 3. EL LOOP (el corazón — leer con atención)

Inspiración: los orquestadores de chat (Claude, etc.) mantienen la
conversación como fuente de verdad y compactan cuando crece. Kapa hace lo
mismo, con una diferencia forzada por el modelo: gemma-4-E2B es un 2B con
ventana de 8192 y **no** sostiene un bucle de herramientas (probado: emitía
las tool-calls como texto). Así que en vez de un agente que decide qué
herramienta llamar, Kapa ejecuta un **plan fijo de 3 llamadas**, cada una con
un solo trabajo. La "inteligencia de agente" está concentrada en la PRIMERA
llamada, que es la única que ve toda la conversación y decide qué hacer.

Principio rector, en una frase:
**El LLM LEE (conversación → intención). El código ESCRIBE (estado) y VERIFICA
(que la respuesta no mienta). El LLM REDACTA (estado → lenguaje natural).**

### 3.0 Puerta de entrada (código, sin LLM)

Antes del loop, dos cosas que NO decide el LLM:

- **¿Debe el bot actuar?** Lo decide el FSM: el debounce dispara solo si el
  comercial no respondió en `RESPONSE_DELAY_MINUTES`. Si respondió → handoff,
  el turno ni empieza.
- **¿Hay algo nuevo que contestar?** Si el único mensaje pendiente es un eco
  del propio bot o no hay mensajes nuevos del cliente → silencio.

Lo que llega al loop: la tanda de mensajes del cliente acumulada en el
debounce (uno o varios), más todo el contexto de la conversación.

### 3.1 Ensamblado de contexto (código, sin LLM)

Se construye UN paquete de contexto, presupuestado, que alimenta la llamada 1:

| Ingrediente | Qué es | Presupuesto aprox. |
|---|---|---|
| Resumen | El resumen compactado de la conversación (§3.6) | ~400 tok |
| Hilo reciente | Las últimas ~10 intervenciones, **etiquetadas por rol**: `CLIENTE:`, `COMERCIAL:`, `KAPA:` | ~900 tok |
| Pedido actual | El carrito, numerado (nombre, cantidad, código) | ~200 tok |
| Pendientes | Preguntas abiertas del bot y opciones vivas | ~200 tok |
| Mensaje(s) del turno | Lo que hay que contestar ahora | ~800 tok |

Los **roles importan**: en la conversación intervienen cliente, comercial y
bot. "Quítame el otro" solo se entiende viendo qué dijo KAPA justo antes;
"como acordamos" puede referirse a algo que dijo el COMERCIAL. El LLM debe ver
quién dijo qué.

Total típico ~2500 tok (30% de la ventana). Nunca se "recorta a posteriori":
si el hilo + resumen se acercan al 60% de la ventana, se dispara la
compactación (§3.6) ANTES, no se truncan mensajes a lo bruto.

### 3.2 Llamada 1 — COMPRENDER (el cerebro del turno)

Única llamada que ve la conversación entera. Entrada: el paquete de §3.1.
Tarea: leer el hilo y decidir qué quiere el cliente **ahora**, resolviendo
referencias a lo anterior. Salida (JSON, un objeto):

```
{
  "hablar": true|false,          # ¿el bot debe responder este turno?
  "derivar": null | "precio" | "especificacion" | "plazo_factura"
             | "queja" | "hablar_comercial" | "otro",
  "actos": [ ... ],              # operaciones de pedido, con referencias TEXTUALES
  "cerrar": true|false,          # el cliente da el pedido por terminado
  "rehacer_anterior": true|false # quiere repetir un pedido pasado
}
```

- `hablar=false` → **silencio** (charla que no necesita al bot, o el comercial
  ya está en ello). Esto es lo que hace que el bot conteste poco.
- `derivar != null` → el bot **no** intenta resolverlo; redacta una derivación
  natural y (según el tipo) hace handoff. `hablar_comercial` → silencio +
  handoff.
- `actos` usa el vocabulario cerrado (pedir / modificar / elegir / rechazar /
  confirmar), cada uno con **evidencia** (el fragmento literal del mensaje) y
  **referencias textuales**, nunca códigos ni índices. Los deícticos ("ese",
  "el segundo", "el otro", "sí ese", "13") se resuelven aquí porque el LLM ve
  contra qué: la pregunta anterior de KAPA está en el hilo.

Un turno normal produce pocos actos. Un turno de charla produce `hablar=false`
y nada más. Una lista larga produce muchos `pedir`.

Presupuesto: 1 llamada. Salida ~400 tok.

### 3.3 Resolución (código + llamada 2 solo si hace falta)

Por cada referencia de producto, el código la resuelve contra el recuperador
del cliente (hoy: su historial K_BOT==0; mañana: catálogo filtrado):

- Se decide **sin LLM** cuando se puede: código tecleado que existe, alias
  exacto, candidato único que cubre lo dicho, cero candidatos, dominancia
  estricta (el producto gana al tester / al material de expositor porque sus
  palabras son un subconjunto), nombres idénticos tras normalizar.
- Solo lo **genuinamente ambiguo** (varios candidatos distinguibles) va a la
  **llamada 2 — MATCH**: una sola llamada por lotes, salida por índice (jamás
  un código nuevo). Tope de ítems por llamada; los excedentes quedan como
  "por aclarar" sin llamar.

Manejo de **no encontrado** y **listas largas** (requisito explícito, §2):

- Si **no** encuentra un producto: se disculpa y pide **marca u otra palabra
  clave**, con honestidad ("no lo veo entre lo que sueles pedir"). Nunca se lo
  inventa.
- Si encuentra **varias** coincidencias: ofrece **código + el factor que las
  diferencia** (p. ej. "15 ml" vs "25 ml"), si lo hay. Si nada las diferencia,
  el código es la única información accionable y se da.
- En **listas de 3 o más**, PROHIBIDO sobrecargar: nada de recitar opciones
  para cada producto ni listas de sí/no. El bot **sabe internamente** cuáles
  no resolvió, pero al cliente le pide ayuda de forma natural — por marca o
  detalle— **sin listárselos todos**, o va de **uno o dos en uno** para que
  los aclare. Lo demás del pedido se apunta y se sigue.

### 3.4 Reductor (código, único escritor)

Toma las operaciones ya resueltas y las aplica al estado. **Es el único punto
del sistema que escribe carrito y pendientes.** Puro, sin I/O. Aquí viven las
garantías estructurales que NO se delegan al LLM porque son aritmética o
integridad:

- Cantidades: nunca se inventan; un decremento a ≤0 pregunta, no borra; una
  línea solo desaparece por borrado explícito; el mismo producto con dos
  cantidades pregunta con los números ya calculados (el LLM no suma).
- Media (imagen/audio/doc) → los productos van a **propuesta a confirmar**,
  no directos al carrito.
- `rehacer_anterior` → propuesta desde el pedido pasado, **nunca** escritura
  directa; requiere confirmación.
- `cerrar` → propuesta de cierre; el `confirmar` sobre ella pasa a CLOSED.
- Toda duda viva produce un "hecho" para que nunca se caiga del mensaje.

El reductor emite una lista de **hechos** tipados (qué pasó: apuntado,
ambiguo, no encontrado, conflicto, propuesta, cerrado, derivado…). No redacta.

### 3.5 Llamada 3 — REDACTAR + VERIFICAR (código de cinturón)

El LLM redacta la respuesta **desde los hechos + la cola de la conversación**
(para tono y continuidad). Es lenguaje natural real, no una plantilla: por eso
suena a persona y no a máquina. Reglas que se le dan: WhatsApp (sin markdown),
breve, no sobrecargar, disculparse si no encontró algo, no hablar de precios
ni prometer plazos.

El **cinturón** es código, no confianza: `verificar()` comprueba sobre el
texto final que (a) todo código que aparezca esté fundado en el estado o el
retrieval del turno, y (b) toda cantidad case con el carrito. Si falla, se
descarta la redacción y sale una plantilla mínima desde el estado (fallback
de seguridad, no el caso normal). Un fallo de verificar es un incidente a
registrar, no un modo de operación.

Presupuesto: 1 llamada.

### 3.6 Resumen / compactación (mantenimiento, no cada turno)

El resumen mantiene el hilo cuando la conversación crece más allá de lo que
cabe en la ventana. Disparo: cuando el contexto acumulado (resumen + mensajes
crudos) alcanza el **60% de la ventana** del LLM (`CIMA_CTX_WINDOW`, 8192 →
~4900 tok). Al dispararse:

1. Se toman el **resumen anterior** + los **mensajes nuevos** aún no resumidos.
2. Una llamada al LLM produce el **resumen nuevo** (qué se ha pedido y
   confirmado, qué quedó pendiente, tono/contexto del cliente).
3. Los mensajes crudos ya resumidos se sueltan del hilo; se conservan solo las
   últimas ~10 intervenciones para el detalle inmediato.

El resumen se persiste en `session.summary`. NO habla de pedidos pasados como
tema; solo guarda el estado de ESTA conversación. Es una llamada de
mantenimiento ocasional, fuera del presupuesto del turno.

### 3.7 Presupuesto total del turno

| Llamada | Cuándo | Coste |
|---|---|---|
| COMPRENDER | siempre | 1 |
| MATCH | solo si hay ambigüedad real | 0-1 |
| REDACTAR | si el bot habla | 0-1 |
| (compactar) | ocasional, al 60% de ventana | fuera de turno |

Techo del turno: **3 llamadas**. Charla que no requiere bot: **1** (COMPRENDER
dice `hablar=false` y se acabó). Pedido simple sin ambigüedad: **2**.

### 3.8 Diagrama del loop

```
mensajes del cliente (debounce ya garantizó: comercial ausente)
        │
   [código] ensamblar contexto: resumen + hilo(roles) + carrito + pendientes
        │
   [LLM 1] COMPRENDER  ── hablar? derivar? actos? cerrar? rehacer?
        │        │
        │        ├─ hablar=false ─────────────► silencio
        │        └─ derivar!=null ────────────► redactar derivación / handoff
        │
   [código] RESOLVER referencias  ──(ambiguo)──►  [LLM 2] MATCH (lotes, índice)
        │
   [código] REDUCTOR (único escritor) ──► estado' + hechos
        │
   [LLM 3] REDACTAR desde hechos + hilo ──► [código] VERIFICAR ──(falla)──► plantilla
        │
   [código] persistir + enviar   (si al 60% ventana: compactar resumen)
```

---

## 4. Qué se ELIMINA del diseño rígido actual

El loop de §3 hace innecesarias, y hay que borrar, las reglas que se
acumularon para tapar la falta de conversación:

- El "salvamento estructural" del número suelto y el parche del "13" en
  resolución → los resuelve COMPRENDER leyendo la pregunta anterior.
- Los dos intentos de extracción y el schema que cima ignora → una sola
  llamada COMPRENDER en modo laxo con el formato en el prompt.
- La plantilla como salida primaria → pasa a ser solo el fallback de
  verificar; la salida normal la redacta el LLM.
- Toda lógica de "elegir/rechazar" que intentaba adivinar el referente sin ver
  el hilo → el hilo está ahora en el prompt.

Lo que SOBREVIVE (es código correcto, no rigidez): el reductor como único
escritor, la resolución determinista (dominancia, nombres idénticos, código
tecleado), el retrieval por historial, `verificar`, el FSM/debounce, el
outbox anti-eco.

---

## 5. Variables de entorno relevantes

| Var | Default | Qué controla |
|---|---|---|
| `RESPONSE_DELAY_MINUTES` | 1 (prod 15) | Espera antes de que el bot cubra al comercial |
| `BOT_SESSION_GAP_HOURS` | 6 | Fin de conversación por silencio |
| `CIMA_CTX_WINDOW` | 8192 | Ventana del LLM; el resumen se dispara al 60% |
| `RAG_HISTORY_ONLY` | true | Resolver solo contra el historial del cliente (hasta limpiar catálogo) |
| `CIMA_MODEL` | gemma-4-E2B Q8_0 | Modelo único (texto/visión/audio/embed) |

---

## 6. Invariantes (qué no puede pasar nunca)

1. El bot habla solo si el comercial no respondió a tiempo.
2. El bot no responde a lo no-comercial; deriva lo comercial que no puede
   atender; calla si piden al comercial.
3. Nunca inventa un código ni una cantidad (garantía de código en
   resolución+verificar; de cantidad en el reductor).
4. Ninguna línea del pedido cambia o desaparece sin que el cliente lo pida.
5. No cierra sin confirmación explícita.
6. En listas largas no sobrecarga: pide ayuda por marca/detalle o va de 1-2 en
   1-2, sin recitar opciones ni listas de sí/no.
7. Mantiene el hilo: entiende referencias a mensajes y pedidos anteriores.
8. Redacta en lenguaje natural; el mensaje nunca contradice el pedido
   (verificar es el cinturón).

---

## 7. Pendiente

- Implementar el loop de §3 (reescritura de la Etapa 1: de extracción ciega a
  COMPRENDER con conversación).
- Recuperar el generador de resumen (§3.6): vivía en el `llm.py` antiguo y se
  perdió en la reescritura; `session.summary` lo espera.
- Ampliar el recuperador al catálogo filtrado cuando K_BOT esté limpio; el
  loop no debe asumir que la fuente es solo el historial.
- Medir en la batería de 001764: fracción de turnos con `hablar=false`
  correcto, referencias resueltas sin preguntar, incidentes de verificar (~0).