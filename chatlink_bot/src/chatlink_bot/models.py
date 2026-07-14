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

    conv_open = Column(Boolean, default=False)
    bot_enabled = Column(Boolean, default=True)
    bot_introduced_at = Column(DateTime(timezone=True), nullable=True)
    last_client_msg_at = Column(DateTime(timezone=True), nullable=True)
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
    K_BOT = Column(SmallInteger)

    PrecioVenta = Column(Numeric)
    StockMinimo = Column(Numeric)
    StockMaximo = Column(Numeric)


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