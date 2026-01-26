# chatlink_bot/src/chatlink_bot/models.py
from enum import Enum as PyEnum
from sqlalchemy import Column, Integer, String, Enum, DateTime, Boolean, Text, Numeric, SmallInteger
from sqlalchemy.sql import func
from pydantic import BaseModel, EmailStr

from .database import PGBase, MSBase


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

    # NEW: “turning on that user for the bot app”
    enabled = Column(Boolean, default=False)

    # NEW: store device jid for safe delete-device on logout
    wa_device_jid = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Chat(PGBase):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(String, index=True, nullable=False)  # Client Phone
    user = Column(String, nullable=False)                 # Salesman Phone
    client = Column(String, nullable=False)               # Client Phone
    message = Column(Text, nullable=True)
    direction = Column(String, nullable=False)            # sent/received
    input_type = Column(Enum(InputType), default=InputType.TEXT, nullable=False)
    is_bot = Column(Boolean, default=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class EmailChat(PGBase):
    __tablename__ = "email_chats"

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(String, index=True, nullable=False)  # Client Email
    user = Column(String, nullable=False)                 # Salesman Email
    client = Column(String, nullable=False)               # Client Email
    message = Column(Text, nullable=True)
    direction = Column(String, nullable=False)
    input_type = Column(Enum(InputType), default=InputType.TEXT, nullable=False)
    is_bot = Column(Boolean, default=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


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
    email: EmailStr
    phone: str
    role: UserRole
    enabled: bool
    wa_device_jid: str | None = None

    class Config:
        from_attributes = True
