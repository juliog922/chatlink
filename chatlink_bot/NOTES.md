# WORKFLOW SPECIFICATION: MULTI-CHANNEL SALES ASSISTANT

## 1. Goal
Build a multi-channel sales order assistant that monitors WhatsApp (WhatsMeow) and Gmail (IMAP/SMTP). It normalizes messages into a single stream and only reacts to known clients found in SQL Server.

## 2. Core Logic
1.  **Ingestion:**
    * **WhatsApp:** Via gRPC (WhatsMeow). Salesmen pair via QR.
    * **Email:** Per-user IMAP monitoring.
    * **Admin Control:** An "Admin" WhatsApp channel listens for commands ("login [email]", "logout [email]") from specific numbers to manage Salesman sessions (generate QR codes, start/stop email monitoring).
2.  **Identification (The Gatekeeper):**
    * Incoming messages are checked against SQL Server `Clientes`.
    * **Phone Match:** Check `Telefono`, `Telefono2`, `Telefono3`.
    * **Email Match:** Check `EMail1`, `EMail2`.
    * *Action:* Discard if no match.
3.  **Content Processing:**
    * Persist all messages (Text + Media).
    * **Audio:** Transcribe via Whisper.
    * **Images:** Extract text via Vision model.
    * **Docs:** Extract text from PDF/CSV/XLSX/DOC/TXT.
4.  **Conversation State Machine:**
    * **Debounce:** 15-minute timer per conversation (reset on new client msg).
    * **Human Takeover:** If commercial replies before timer ends, cancel event.
    * **Trigger:** When timer fires, load incremental history.
5.  **Order Intelligence (RAG & LLM):**
    * **Product Indexing (Qdrant):** On startup + Daily Event. Indexes `DescripcionArticulo`, `Descripcion2Articulo`, `DescripcionLinea`, `ComentarioArticulo`, `MarcaProducto` where `K_BOT=1`.
    * **Summarizer:** Analyze history for (1) Closed Orders, (2) Building Orders, (3) Mentioned Products, (4) Narrative Context.
    * **Hybrid Search:** Split client messages into intent blocks. Run hybrid search (Dense + Sparse) on Qdrant to find potential products.
    * **LLM Context:** Send Summary + Top K Product Candidates + Recent Chat.
    * **LLM Role:** Strictly order building. Clarify ambiguities. Natural but professional tone.
    * **Non-Commercial Guardrail:** If message is purely conversational/status check, respond: "The commercial will contact you soon."
6.  **Output:**
    * Confirmed list -> Generate XLSX -> Send to Commercial (via Admin Gmail).

---

# DATABASE SCHEMA REFERENCE (SQL SERVER)
*Read-Only access used for Client Identification and Product Lookup.*

## TABLE: Clientes
**Primary Key:** `CodigoCliente` (assumed based on usage)

### RELEVANT COLUMNS (Used for Identification & Context)
| Column | Type | Description |
| :--- | :--- | :--- |
| **Nombre** | varchar(35) | Client Name (e.g., "PELAYO OSUNA, LAURA") |
| **Telefono** | varchar(15) | Primary Phone (e.g., "685176889") |
| **Telefono2** | varchar(15) | Secondary Phone |
| **Telefono3** | varchar(15) | Tertiary Phone |
| **EMail1** | varchar(250) | Primary Email (e.g., "administracion@kapalua.es") |
| **EMail2** | varchar(250) | Secondary Email |

### OTHER COLUMNS (Context / Informational)
* **Location:** `Domicilio`, `CodigoPostal`, `Municipio`, `Provincia`, `SiglaNacion`.
* **Business:** `CifDni` (Tax ID), `RazonSocial`, `FormadePago`, `TarifaPrecio`.
* **Logistics:** `CodigoTransportista`, `ObservacionesCliente`.
* **Metadata:** `FechaAlta`, `CodigoEmpresa`, `CodigoCliente`.
* **Legacy/Internal:** `ActivarLogicNet`, `UsuarioLogicNet`, `K_DtoVolumen`, `KNumTiendas`.

## TABLE: Articulos (Mapped to Products)
**Primary Key:** `CodigoArticulo`

### RELEVANT COLUMNS (Used for Order Resolution)
| Column | Type | Description |
| :--- | :--- | :--- |
| **CodigoArticulo** | varchar(21) | Unique SKU (e.g., "00104") |
| **DescripcionArticulo** | varchar(60) | Main Name (e.g., "GN QUITA ESMALTE ROS...") |
| **Descripcion2Articulo** | varchar(40) | Secondary Name |
| **DescripcionLinea** | text | Extended Description |
| **ComentarioArticulo** | text | Internal comments |
| **MarcaProducto** | varchar(50) | Brand (e.g., "GENA", "TAIFF") |
| **K_BOT** | smallint | Bot Visibility Flag (e.g., -1 or 1) |

### OTHER COLUMNS (Context / Informational)
* **Pricing:** `PrecioVenta`, `PrecioCosteEstandar`, `%Descuento`, `IvaIncluido`.
* **Stock:** `StockMinimo`, `StockMaximo`, `PuntoPedido`.
* **Logistics:** `PesoBrutoUnitario_`, `KUnidXCaja`, `KEANCaja`.
* **Dimensions:** `KAltoArticulo`, `KAnchoArticulo`, `KLargoArticulo`.
* **Classification:** `CodigoFamilia`, `CodigoSubfamilia`, `Temporada`.

---

# DATABASE SCHEMA REFERENCE (POSTGRESQL)
*Write access. Stores Application State, Users, and Normalized Chat History.*

## TABLE: app_users
*Stores internal system users (Salesmen/Admins).*

| Column | Type | Description |
| :--- | :--- | :--- |
| **id** | Integer (PK) | Auto-incrementing ID. |
| **name** | String | Full name of the user. |
| **email** | String (Unique) | User email (used for login/notifications). |
| **phone** | String | User phone number. |
| **role** | Enum | 'admin' or 'user'. |
| **created_at** | DateTime | Record creation timestamp. |

## TABLE: chats
*Stores normalized WhatsApp messages.*

| Column | Type | Description |
| :--- | :--- | :--- |
| **id** | Integer (PK) | Auto-incrementing ID. |
| **chat_id** | String | The unique identifier for the conversation (Client Phone). |
| **user** | String | The internal user (Salesman) phone involved. |
| **client** | String | The client phone involved. |
| **message** | Text | The content of the message. |
| **direction** | String | 'sent' (Salesman->Client) or 'received' (Client->Salesman). |
| **input_type** | Enum | 'text', 'image', 'audio', 'pdf', 'xlsx', etc. |
| **is_bot** | Boolean | True if the message was generated by the AI agent. |
| **timestamp** | DateTime | When the message occurred. |
| **created_at** | DateTime | When the record was saved. |

## TABLE: email_chats
*Stores normalized Email threads.*

| Column | Type | Description |
| :--- | :--- | :--- |
| **id** | Integer (PK) | Auto-incrementing ID. |
| **chat_id** | String | The unique identifier (Client Email). |
| **user** | String | The internal user (Salesman) email. |
| **client** | String | The client email. |
| **message** | Text | Content (Subject + Body). |
| **direction** | String | 'sent' or 'received'. |
| **input_type** | Enum | 'text', 'pdf', 'xlsx', etc. (based on attachment). |
| **is_bot** | Boolean | True if the email was generated by the AI agent. |
| **timestamp** | DateTime | When the email was sent/received. |
| **created_at** | DateTime | When the record was saved. |

---

# DEVELOPMENT PLAN

## 1. Logic Modules & Responsibilities

### A. **Gateway Module (Ingestion & Normalization)**
* **Role:** The "Ears" of the system.
* **Responsibilities:**
    * Manage gRPC connection to `chatlink-meow` (WhatsApp).
    * Manage multi-threaded IMAP listeners (Gmail).
    * Admin Command Listener: Monitor "Admin" WhatsApp for commands (`login`, `logout`) to spawn QR codes or kill sessions.
    * **Crucial:** Filter traffic. Check every incoming ID (Phone/Email) against SQL Server. If not found -> Drop immediately.
    * Normalize valid messages into `MessageEvent` objects and persist to Postgres (`chats` / `email_chats`).

### B. **State Machine Module (The Orchestrator)**
* **Role:** The "Clock" and "State Keeper".
* **Responsibilities:**
    * Maintain a per-conversation state: `IDLE` -> `ACTIVE` -> `DEBOUNCING` -> `PROCESSING`.
    * **Debounce Logic:** On client msg, set timer (15m). If commercial replies, Cancel Timer (Human Takeover).
    * **Trigger:** When timer expires, trigger the **Intelligence Module**.

### C. **Intelligence Module (RAG & Parsing)**
* **Role:** The "Brain".
* **Responsibilities:**
    * **Text Extraction:** Route media (Images/Docs) to specific parsers/OCR.
    * **Intent Splitting:** Pre-process client text to isolate potential product mentions from chatter.
    * **Vector Search (Qdrant):**
        * **Startup/Daily:** Index `Articulos` (where `K_BOT=1`) into Qdrant. Fields: `Descripcion*`, `Comentario`, `Marca`.
        * **Runtime:** Perform Hybrid Search (Dense + Keyword) for each extracted intent block. Return `Top-K` candidates with scores.
    * **LLM Integration:** Assemble context (Summary + Candidates + Chat) and call LLM.

### D. **Output Module (Action)**
* **Role:** The "Hands".
* **Responsibilities:**
    * **Response Generation:** Send LLM response to Client (WhatsApp/Email).
    * **Order Finalization:** If LLM detects "Confirmed Order":
        * Generate structured XLSX.
        * Send Email to Commercial (using Admin Gmail credentials) with XLSX attachment.

---

## 2. Conversation State Machine (Finite State Machine)

We will implement a simplified FSM for each Client-Salesman pair.

1.  **IDLE**: No recent activity.
    * *Event:* Client Message -> Transition to **DEBOUNCING**. Start 15m Timer.
2.  **DEBOUNCING**: Waiting for potential follow-up messages.
    * *Event:* Client Message -> Reset 15m Timer. Stay in **DEBOUNCING**.
    * *Event:* Commercial Message -> **CANCEL**. Transition to **IDLE** (Human replied).
    * *Event:* Timer Expired -> Transition to **PROCESSING**.
3.  **PROCESSING**: AI is working.
    * *Action:* Run RAG -> Call LLM -> Send Reply.
    * *Transition:* Back to **IDLE**.

---

## 3. Qdrant & RAG Strategy

* **Collection Name:** `products`
* **Payload Fields:**
    * `CodigoArticulo` (ID)
    * `DescripcionArticulo`
    * `MarcaProducto`
    * `K_BOT`
* **Indexing Strategy:**
    * Combine `DescripcionArticulo` + `Descripcion2Articulo` + `DescripcionLinea` + `MarcaProducto` into a single text blob for embedding.
    * Use a Multi-Lingual model (e.g., `paraphrase-multilingual-mpnet-base-v2`) for dense vectors.
    * Enable Sparse Vectors (BM25) for exact keyword matching (critical for specific brand names or codes).

---

## 4. LLM Prompt Proposals (Provisional)

### Prompt A: The Summarizer (Internal Logic)
*Goal: Maintain a running summary of the order status AND the conversational context.*

```text
You are an Order Assistant. Update the conversation summary based on the new messages.

Current Summary: {current_summary}
New Messages:
{new_messages_text}

Output JSON:
{
  "order_status": "BUILDING" | "CLOSED" | "IDLE",
  "confirmed_items": [{"code": "...", "qty": 1}],
  "potential_mentions": ["list", "of", "vague", "products"],
  "last_interaction_intent": "ORDER" | "QUESTION" | "CHATTER",
  "chat_context_summary": "Brief narrative of the conversation flow so far. (e.g., 'User asked for shampoo prices, rejected the first offer, and is now asking for conditioners. User seems in a hurry.')"
}
```

### Prompt B: The Order Builder (The Client Response)
*Goal: Clarify products or confirm order with a NATURAL, helpful tone.*

```text
You are a Sales Assistant for a Cosmetics Distributor.
Your Goal: Help the client build an order efficiently.

Context:
- Client Name: {client_name}
- Conversation Narrative: {chat_context_summary}
- Current Order State: {confirmed_items}
- Found Product Candidates (from Database):
  {rag_candidates_json} (Format: "User Text" -> [Top 3 DB Matches with Confidence])

Rules:
1. ONLY discuss the order. If the user asks about prices, delivery times, or personal topics, politely say: "The commercial will contact you soon regarding those details."
2. If the user mentions a product vaguely, use the Candidates list to propose the most likely match. Ask for confirmation naturally.
3. If multiple candidates are similar, list them clearly (Name + Brand) and ask which one they prefer.
4. If the user confirms an item, acknowledge it briefly and move to the next point.
5. If the order seems complete, ask for final confirmation of the full list.
6. **TONE:** Speak Spanish (or user language). Be professional and natural. Avoid sounding robotic. Do not engage in unnecessary small talk, but maintain a polite, helpful flow.

User Last Message: "{last_message}"

Response:
```

## 5. Architecture Overview: Event-Driven Modular Monolith

The design strictly separates "listening" (Transport), "thinking" (AI/Logic), and "data" (Database). The Event Bus (events.py) is the spine of the application—modules do not call each other directly; they emit events or subscribe to them.

1. Transport (Ingestion): Listens to WhatsApp/Email, normalizes data, and emits message_received.
2. Logic (Orchestrator): Subscribes to events, manages the Conversation State Machine (Debounce), and decides when to trigger the AI.
3. AI (Intelligence): Pure functions that take context and return answers (RAG + LLM).

```text
chatlink_bot/
├── src/
│   └── chatlink_bot/
│       ├── __init__.py
│       ├── main.py              # Entry point: Wires services, starts API & listeners
│       ├── config.py            # Centralized environment variables
│       ├── events.py            # The Event Bus (Pub/Sub pattern)
│       ├── database.py          # Async Engines for Postgres (Write) & SQL Server (Read)
│       ├── models.py            # SQLAlchemy Tables (Users, Chats, Clientes, Articulos)
│       │
│       ├── whatsapp_pb2.py      # Protobuf Generated File
│       ├── whatsapp_pb2_grpc.py # gRPC Generated File
│       │
│       ├── transport/           # "The Ears & Hands" (Input/Output)
│       │   ├── __init__.py
│       │   ├── whatsapp.py      # Wrapper for whatsmeow gRPC. Imports pb2 files from parent.
│       │   └── email.py         # IMAP Listener & SMTP Sender
│       │
│       ├── logic/               # "The Orchestrator" (State Machine)
│       │   ├── __init__.py
│       │   ├── fsm.py           # Finite State Machine (Idle -> Debouncing -> Processing)
│       │   └── handlers.py      # Event Consumers: Connects Transport -> FSM -> AI
│       │
│       ├── ai/                  # "The Brain" (Intelligence)
│       │   ├── __init__.py
│       │   ├── rag.py           # HybridRetriever logic (your provided code)
│       │   ├── qdrant.py        # Qdrant Client setup & Indexing logic
│       │   ├── llm.py           # LLM Client & Prompt Templates (Summarizer/Builder)
│       │   └── parsers.py       # Whisper, Vision, & Doc extraction logic
│       │
│       └── api/                 # "The Control Panel" (FastAPI)
│           ├── __init__.py
│           └── routes.py        # Endpoints for User Mgmt, QR generation, Health
│
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Architecture & File Logic Description
This architecture follows an Event-Driven Modular Monolith pattern. The core idea is to decouple "Listening" (Transport), "Thinking" (AI), and "Orchestrating" (Logic) using a central Event Bus.

1. Core Infrastructure
- events.py: The "spine" of the application. It defines the EventBus class. Modules communicate by emitting events (e.g., message_received) rather than importing and calling each other directly.

- database.py: Handles database connectivity.

    - Postgres (AsyncSessionPG): For write-heavy operations (Chat History, User State).
    - SQL Server (AsyncSessionSQL): For read-only access to the legacy ERP (Clientes, Articulos).


- models.py: Defines all SQLAlchemy ORM models.

    - Application State: User, Chat, EmailChat.
    - Legacy Mirrors: MSClient, MSArticle.

2. Transport (Ingestion & Output)
- transport/whatsapp.py:

    - Logic: Acts as a wrapper for the whatsapp_pb2 generated files.
    - Responsibility: Manages the gRPC StreamMessages loop. It normalizes incoming gRPC events into a standard dictionary and emits event_bus.emit('message_received', data). It also handles outgoing commands like send_message.

- transport/email.py:

    - Logic: Runs a multi-threaded IMAP listener.
    - Responsibility: Polls for unseen emails, parses attachments/body, and emits event_bus.emit('email_received', data).

3. Logic (The Orchestrator)
- logic/fsm.py:

    - Logic: The "Clock" and State Machine.
    - Responsibility: Manages the 15-minute Debounce Timer per conversation.
    - Start/Reset: On new client message.
    - Cancel: On commercial reply (Human Takeover).
    - Trigger: When the timer expires, it emits trigger_ai_processing.

- logic/handlers.py:

    - Logic: The "Controller".
    - Responsibility: Subscribes to events from the bus.
        - on_message_received: Validates the sender against SQL Server (Identification). If valid, saves the chat and calls fsm.start_timer().
        - on_ai_trigger: Orchestrates the AI flow (Load History -> RAG -> LLM -> Send Reply).

4. AI (The Brain)
- ai/rag.py:

    - Logic: Implements the HybridRetriever logic.
    - Responsibility: Performs Dense (Vector) + Sparse (BM25) search on Qdrant, followed by Cross-Encoder Reranking to find the best product matches.

- ai/qdrant.py:

    - Logic: Qdrant connection management.
    - Responsibility: Runs the Indexing Routine on startup/daily events to sync SQL Server Articulos (where K_BOT=1) into the Vector Database.

- ai/llm.py:

    - Logic: LLM Client and Prompt Engineering.
    - Responsibility: Contains the Summarizer and Order Builder prompts. It assembles the context (Chat History + RAG Candidates) and calls the LLM provider.

- ai/parsers.py:

    - Logic: File processing.
    - Responsibility: Extracts text from non-text inputs (Whisper for Audio, Vision for Images, Text extractors for PDF/Docs) before persistence.

5. API (Control)
- api/routes.py:

    - Logic: FastAPI Endpoints.
    - Responsibility: Handles administrative tasks like User Registration, initiating WhatsApp Login (QR Code generation), and System Health Checks.
