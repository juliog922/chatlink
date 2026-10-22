# chatlink_bot/src/chatlink_bot/models.py
from enum import Enum as PyEnum

from pydantic import BaseModel, EmailStr
from sqlalchemy import (
    JSON, Boolean, Column, DateTime, Enum, Integer, Numeric, SmallInteger,
    String, Text, UniqueConstraint,
)
from sqlalchemy.sql import func

from .database import MSBase, PGBase


# --- Enums ---
class UserRole(str, PyEnum):
    ADMIN = "admin"
    USER = "user"


class InputType(str, PyEnum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    PDF = "pdf"
    XLSX = "xlsx"
    CSV = "csv"
    DOC = "doc"
    DOCX = "docx"
    UNKNOWN = "unknown"


# --- Postgres Tables (Write) ---
class User(PGBase):
    __tablename__ = "app_users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    phone = Column(String, nullable=False)
    role = Column(Enum(UserRole), default=UserRole.USER)
    enabled = Column(Boolean, default=False)          # bot service on/off for this salesman
    wa_device_jid = Column(String, nullable=True)     # for safe delete-device on logout
    imap_uids = Column(JSON, default=dict)            # {folder: last UID seen}, survives restarts
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    @property
    def has_gmail_password(self) -> bool:
        from .transport.email import email_transport
        return email_transport.has_app_password(self.email)


class Chat(PGBase):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(String, index=True, nullable=False)  # Client phone
    user = Column(String, nullable=False)                 # Salesman phone
    client = Column(String, nullable=False)               # Client phone
    message = Column(Text, nullable=True)
    direction = Column(String, nullable=False)            # sent | received
    input_type = Column(Enum(InputType), default=InputType.TEXT, nullable=False)
    is_bot = Column(Boolean, default=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # Bot replies are written "queued" in the turn's transaction and sent by
    # the loop: queued -> sent | failed (after SEND_ATTEMPTS tries).
    status = Column(String, default="sent")
    attempts = Column(Integer, default=0)


class EmailChat(PGBase):
    __tablename__ = "email_chats"

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(String, index=True, nullable=False)  # Client email
    user = Column(String, nullable=False)                 # Salesman email
    client = Column(String, nullable=False)               # Client email
    message = Column(Text, nullable=True)
    direction = Column(String, nullable=False)
    input_type = Column(Enum(InputType), default=InputType.TEXT, nullable=False)
    is_bot = Column(Boolean, default=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # Bot replies are written "queued" in the turn's transaction and sent by
    # the loop: queued -> sent | failed (after SEND_ATTEMPTS tries).
    status = Column(String, default="sent")
    attempts = Column(Integer, default=0)


class ConversationSession(PGBase):
    """
    Persistent per-conversation state, keyed by (channel, client, salesman).

    Replaces the old in-memory `_state` dict in ai/llm.py (keyed by client
    only, lost on restart). It is the deterministic home for the conversation
    LIFECYCLE, which the LLM must obey but never decide:

      conv_open          False = the conversation ENDED (order dispatched, or
                         the client went silent past BOT_SESSION_GAP_HOURS).
                         The next COMMERCIAL intent starts a new conversation
                         (fresh cart + self-introduction); pleasantries alone
                         keep it ended. Set back to True only by real order
                         activity.
      bot_enabled        Hard opt-out: bot never replies here when False.
      bot_introduced_at  When Kapa last introduced itself (None = never).
      last_closed_cart   Recap of the last DISPATCHED order — context for
                         "ponme lo mismo que la última vez" and grounding for
                         repeat-order add_item calls.
    """
    __tablename__ = "conversation_sessions"
    __table_args__ = (UniqueConstraint("channel", "client_id", "user_id"),)

    id = Column(Integer, primary_key=True)
    channel = Column(String, nullable=False)              # whatsapp | email
    client_id = Column(String, index=True, nullable=False)
    user_id = Column(String, nullable=False)              # salesman phone/email

    order_status = Column(String, default="IDLE")         # IDLE | BUILDING | CLOSED
    cart = Column(JSON, default=list)                     # [{"code": str, "qty": int}]
    summary = Column(Text, default="")                    # rolling context note
    last_closed_cart = Column(JSON, default=list)         # last dispatched order
    # Multi-item work queue: unresolved products with their option snapshots,
    # or estado="enriquecer" items (Pass-1 AMBIGUOUS) awaiting client detail.
    open_items = Column(JSON, default=list)
    # One-time "how I work best" capability guidance delivered flag.
    guide_shown = Column(Boolean, default=False)

    # Highest client message id already handled (answered, or deliberately
    # left silent). Messages above it, and above the salesman's last message,
    # are pending: see logic/loop.py.
    last_answered_id = Column(Integer, default=0)

    conv_open = Column(Boolean, default=False)
    bot_enabled = Column(Boolean, default=True)
    bot_introduced_at = Column(DateTime(timezone=True), nullable=True)
    last_client_msg_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class KapaOrder(PGBase):
    """An order Kapa closed. Written in the same transaction as the close;
    the loop emails it to the salesman and retries until it is delivered."""
    __tablename__ = "kapa_orders"

    id = Column(Integer, primary_key=True)
    channel = Column(String, nullable=False)
    client_id = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    client_name = Column(String, default="")
    cart = Column(JSON, default=list)       # [{code, qty, name, fuente}]
    review = Column(JSON, default=list)     # [{texto, qty}] the salesman must resolve
    closed_at = Column(DateTime(timezone=True), server_default=func.now())
    email_status = Column(String, default="queued")   # queued | sent | failed
    attempts = Column(Integer, default=0)


class ClientAlias(PGBase):
    """The client's own words for a product, saved when he picked it."""
    __tablename__ = "client_aliases"

    codigo_cliente = Column(String(15), primary_key=True)
    alias = Column(String, primary_key=True)          # normalize_text() of his words
    code = Column(String(32), nullable=False)
    name = Column(String, default="")
    uses = Column(Integer, default=1)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# --- SQL Server Tables (Read-only mapping) ---
class MSClient(MSBase):
    __tablename__ = "Clientes"

    CodigoCliente = Column(String(15), primary_key=True)
    CodigoEmpresa = Column(SmallInteger, primary_key=True)

    Nombre = Column(String(35))
    Telefono = Column(String(15))
    Telefono2 = Column(String(15))
    Telefono3 = Column(String(15))
    EMail1 = Column(String(250))
    EMail2 = Column(String(250))

    Domicilio = Column(String(40))
    CodigoPostal = Column(String(8))
    Municipio = Column(String(25))
    Provincia = Column(String(20))
    CifDni = Column(String(13))


class MSArticle(MSBase):
    __tablename__ = "Articulos"

    CodigoArticulo = Column(String(21), primary_key=True)
    CodigoEmpresa = Column(SmallInteger, primary_key=True)

    DescripcionArticulo = Column(String(60))
    Descripcion2Articulo = Column(String(40))
    DescripcionLinea = Column(Text)
    ComentarioArticulo = Column(Text)
    MarcaProducto = Column(String(50))
    K_BOT = Column(SmallInteger)                 # -1 = vendible; 0 = inactivo (¡corregido!)

    # Campos para el filtro de corpus y el dedup por reinscripción de EAN.
    CodigoAreaCompetenciaLc = Column(String(10))  # 'MERCH' = merchandising del ERP
    CodigoAlternativo = Column(String(40))        # EAN; agrupa referencias gemelas
    FechaAlta = Column(DateTime)                   # alta del artículo; "el más nuevo"

    PrecioVenta = Column(Numeric)
    StockMinimo = Column(Numeric)
    StockMaximo = Column(Numeric)


# --- Filtro de corpus: UNA definición, un sitio (ver NOTES §3.3.1) ---------
# La usan la ingesta de Qdrant (ai/qdrant.py) y el cargador de historial
# (ai/rag.py). Es un predicado sobre una FILA de artículo, no un WHERE, para
# poder aplicarlo también al leer de las cachés (Postgres / simulación), no
# solo al poblarlas. Cambia el criterio aquí y en un solo sitio.
#
# Reglas (idénticas en historial y catálogo):
#   K_BOT = -1                          -> activo/vendible
#   CodigoAreaCompetenciaLc <> 'MERCH'  -> flag de merchandising del ERP
#   DescripcionArticulo no vacía
#   la descripción NO contiene ninguna palabra de _CORPUS_STOPWORDS
# El campo del ERP solo no basta (deja ~360 en PRO/RET y sin área); palabras
# solas tampoco. Campo + palabras.
K_BOT_VENDIBLE = -1

_CORPUS_STOPWORDS = ("LOTE", "INTRO DEAL", "BONIF", "PROMO",
                     "EXPOSITOR", "DISPLAY", "TESTER", "OBSEQUIO")


def es_vendible(codigo_area=None, descripcion=None, k_bot=None) -> bool:
    """True si el artículo es producto vendible por el bot. Acepta los campos
    sueltos (así lo llaman tanto una fila ORM como una fila de caché)."""
    try:
        if int(k_bot if k_bot is not None else 0) != K_BOT_VENDIBLE:
            return False
    except (TypeError, ValueError):
        return False
    if (codigo_area or "").strip().upper() == "MERCH":
        return False
    desc = (descripcion or "").strip()
    if not desc:
        return False
    up = desc.upper()
    return not any(w in up for w in _CORPUS_STOPWORDS)


def es_vendible_articulo(a) -> bool:
    """Conveniencia para una fila MSArticle."""
    return es_vendible(getattr(a, "CodigoAreaCompetenciaLc", None),
                       getattr(a, "DescripcionArticulo", None),
                       getattr(a, "K_BOT", None))


# --- Pydantic Schemas ---
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    phone: str
    role: UserRole = UserRole.USER


class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    phone: str
    role: UserRole
    enabled: bool
    wa_device_jid: str | None = None
    has_gmail_password: bool = False

    class Config:
        from_attributes = True


# --- SQL Server: order tables (read-only mapping) ---
# Only the columns actually used are mapped, so a SELECT never drags the ~180
# remaining columns of LineasPedidoCliente across the VPN.
#
# The declared PK is the document's natural composite. SQLAlchemy only needs it
# to identify rows; the physical PK of the lines table is LineasPosicion
# (uniqueidentifier), which we do not need. MSBase is never passed to
# create_all, so these mappings emit no DDL against SAGE.
class MSOrderHeader(MSBase):
    __tablename__ = "CabeceraPedidoCliente"

    CodigoEmpresa = Column(SmallInteger, primary_key=True)
    EjercicioPedido = Column(SmallInteger, primary_key=True)
    SeriePedido = Column(String(10), primary_key=True)
    NumeroPedido = Column(Integer, primary_key=True)

    CodigoCliente = Column(String(15), index=True)
    FechaPedido = Column(DateTime)
    NumeroLineas = Column(Integer)
    # Browsing-only columns: the order list is unreadable without a name and a
    # total. Both are safe types to map for reads.
    RazonSocial = Column(String(60))
    ImporteLiquido = Column(Numeric)


class MSOrderLine(MSBase):
    __tablename__ = "LineasPedidoCliente"

    CodigoEmpresa = Column(SmallInteger, primary_key=True)
    EjercicioPedido = Column(SmallInteger, primary_key=True)
    SeriePedido = Column(String(10), primary_key=True)
    NumeroPedido = Column(Integer, primary_key=True)
    Orden = Column(Integer, primary_key=True)

    CodigoArticulo = Column(String(21))
    # Descriptions FROZEN at order time: they can differ from today's catalog,
    # and they are extra names by which this client may refer to the product.
    DescripcionArticulo = Column(String(60))
    Descripcion2Articulo = Column(String(40))
    # The reference the CLIENT uses for the article. Where populated, this is a
    # per-client alias that already exists inside SAGE.
    CodigodelCliente = Column(String(30))
    UnidadesPedidas = Column(Numeric)
    Precio = Column(Numeric)
    ImporteNeto = Column(Numeric)


# --- Postgres: per-client order-history cache ---
class ClientProductHistory(PGBase):
    """One article a client has ordered, aggregated.

    Refreshed by full replacement per client: the per-client volume is small
    (tens to a few hundred rows), which removes any need to reason about
    partial diffs.
    """
    __tablename__ = "client_product_history"

    codigo_cliente = Column(String(15), primary_key=True)
    codigo_articulo = Column(String(32), primary_key=True)

    descripcion = Column(Text, default="")               # current catalog name
    descripciones_historicas = Column(JSON, default=list)
    marca = Column(String, default="")
    ref_cliente = Column(String, default="")             # CodigodelCliente
    bot_visible = Column(Boolean, default=True)          # K_BOT == 0

    veces = Column(Integer, default=0)                   # distinct orders
    unidades_total = Column(Numeric, default=0)
    qty_moda = Column(Integer, default=1)                # this client's usual qty
    primera_fecha = Column(DateTime, nullable=True)
    ultima_fecha = Column(DateTime, nullable=True)
    refreshed_at = Column(DateTime(timezone=True), server_default=func.now())


class ClientHistorySync(PGBase):
    """Per-client sync marker.

    Exists because "zero rows" is ambiguous: this is what tells
    'never synced' apart from 'client has no orders'.
    """
    __tablename__ = "client_history_sync"

    codigo_cliente = Column(String(15), primary_key=True)
    refreshed_at = Column(DateTime(timezone=True), nullable=True)
    attempted_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String, default="ok")                # ok | empty | error
    n_products = Column(Integer, default=0)
    n_lines = Column(Integer, default=0)
    n_orders = Column(Integer, default=0)
    n_excluded = Column(Integer, default=0)
    n_ref_cliente = Column(Integer, default=0)
    tolerant_join = Column(Boolean, default=False)
    error = Column(Text, default="")