from sqlalchemy import Column, String, Integer, or_, cast, and_, func
from sqlalchemy.orm import Session
from typing import Optional, List

from src.models import Base_sqlserver


class Articulo(Base_sqlserver):
    __tablename__ = "Articulos"

    codigo = Column("CodigoArticulo", String, primary_key=True)
    descripcion1 = Column("DescripcionArticulo", String)

    # Estos campos debes tenerlos en el modelo si los usas para el filtro:
    codigo_empresa = Column("CodigoEmpresa", Integer)
    obsoleto = Column("ObsoletoLc", String)
    bloqueo_pedido_compra = Column("BloqueoPedidoCompra", String)
    bloqueo_compra = Column("BloqueoCompra", String)

    @staticmethod
    def filtros_basura():
        return and_(
            Articulo.codigo_empresa == 1,
            Articulo.obsoleto == "0",
            Articulo.bloqueo_pedido_compra == "0",
            Articulo.bloqueo_compra == "0",
        )

    @staticmethod
    def get_by_codigo(session: Session, codigo: str) -> Optional["Articulo"]:
        codigo_norm = str(codigo).strip().upper()
        codigo_sin_zeros = codigo_norm.lstrip("0")

        candidatos = (
            session.query(Articulo)
            .filter(
                func.upper(func.ltrim(func.rtrim(Articulo.codigo))).like(
                    f"%{codigo_sin_zeros}"
                ),
                Articulo.filtros_basura(),
            )
            .all()
        )

        for art in candidatos:
            cod = art.codigo.strip().lstrip("0").upper()
            if cod == codigo_norm or cod == codigo_sin_zeros:
                return art

        return None

    @staticmethod
    def get_by_words_list(session: Session, palabras: List[str]) -> List["Articulo"]:
        condiciones = []
        for palabra in palabras:
            like = f"%{palabra}%"
            condiciones.extend(
                [
                    cast(Articulo.descripcion1, String).ilike(like),
                ]
            )
        return (
            session.query(Articulo)
            .filter(or_(*condiciones), Articulo.filtros_basura())
            .all()
        )
