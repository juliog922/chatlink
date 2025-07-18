from sqlalchemy import ForeignKey, DateTime
from sqlalchemy import Column, Integer, String

from src.models import Base_sqlite


class Order(Base_sqlite):
    __tablename__ = "orders"

    id = Column("Id", Integer, primary_key=True)
    codigo_empresa = Column("CodigoEmpresa", Integer, nullable=False)
    codigo_cliente = Column("CodigoCliente", Integer, nullable=False)
    codigo_articulo = Column("CodigoArticulo", String, nullable=False)
    units = Column("units", Integer, nullable=False)
    timestamp = Column("timestamp", DateTime, nullable=False)

    __table_args__ = (
        ForeignKey("Clientes.CodigoCliente", onupdate="CASCADE", ondelete="CASCADE"),
        ForeignKey("Articulos.CodigoArticulo", onupdate="CASCADE", ondelete="RESTRICT"),
    )
