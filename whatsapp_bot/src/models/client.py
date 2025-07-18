from sqlalchemy import Column, Integer, String, or_
from sqlalchemy.orm import Session
from typing import Optional
import logging

from src.models import Base_sqlserver

class Cliente(Base_sqlserver):
    __tablename__ = "Clientes"  # Cambia si el nombre real de la tabla es distinto

    codigo_empresa = Column("CodigoEmpresa", Integer, primary_key=True)
    codigo_cliente = Column("CodigoCliente", Integer, primary_key=True)
    razon_social = Column("RazonSocial", String)
    domicilio = Column("Domicilio", String)
    documento = Column("CifDni", String)

    telefono1 = Column("Telefono", String)
    telefono2 = Column("Telefono2", String)
    telefono3 = Column("Telefono3", String)

    email1 = Column("EMail1", String)
    email2 = Column("EMail2", String)

    @staticmethod
    def get_by_codigo(session: Session, codigo_cliente: int) -> Optional["Cliente"]:
        if session.bind.dialect.name == "sqlite":
            logging.warning("Intento de uso de Cliente.get_by_codigo con SQLite")
            return None
        return (
            session.query(Cliente)
            .filter(Cliente.codigo_cliente == codigo_cliente)
            .first()
        )

    @staticmethod
    def get_by_telefono(session: Session, telefono: str) -> Optional["Cliente"]:
        # 👇 Trampa temporal: número simulado
        if telefono.endswith("688773722"):
            cliente_fake = Cliente(
                codigo_empresa=1,
                codigo_cliente=9998,
                razon_social="Cliente Ficticio",
                domicilio="Calle Falsa 123",
                documento="12345678A",
                telefono1="688773722",
                telefono2=None,
                telefono3=None,
                email1="rengifoivana@gmail.com",
                email2=None,
            )
            return cliente_fake

        like_pattern = f"%{telefono}"
        return (
            session.query(Cliente)
            .filter(
                or_(
                    Cliente.telefono1.ilike(like_pattern),
                    Cliente.telefono2.ilike(like_pattern),
                    Cliente.telefono3.ilike(like_pattern),
                )
            )
            .first()
        )
