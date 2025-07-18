from typing import List, Tuple, Optional
from sqlalchemy.orm import Session
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
from PIL import Image
from openpyxl import Workbook
import logging
import os
import re

from src.models.product import Articulo
from src.models.message import Message
from src.media.sftp import find_image_file, build_order_image_table

def update_order(
    session: Session, productos: List[Tuple[str, str]]
) -> Optional[Image.Image]:
    """
    Generates a visual order summary as an image (with thumbnails from SFTP).

    Args:
        session: SQLAlchemy session.
        productos: List of tuples (codigo, cantidad).

    Returns:
        PIL Image object representing the order summary, or None if no products.
    """
    if not productos:
        logging.warning("No products provided.")
        return None

    items = []

    for codigo, cantidad in productos:
        articulo = Articulo.get_by_codigo(session, codigo)
        descripcion = (
            articulo.descripcion1 if articulo else "Sin coincidencia de Articulos"
        )
        img_bytes = find_image_file(codigo)
        items.append((codigo, cantidad or "", descripcion, img_bytes))

    return build_order_image_table(items)


def confirmed_order(messages: List["Message"]) -> Optional[str]:
    """
    Busca el último mensaje del comercial con un pedido (prefijo 'Pedido:')
    seguido por una confirmación del cliente ('es correcto').
    """

    pedido_idx = None
    confirmacion_idx = None

    logging.info("Iniciando búsqueda de confirmación 'es correcto'...")

    # Buscar confirmación del cliente (último "es correcto")
    for idx in range(len(messages) - 1, -1, -1):
        direction = messages[idx].direction
        content = messages[idx].content
        logging.info(
            f"Revisando mensaje [{idx}] direction={direction} content={repr(content)}"
        )
        if (
            direction == "received"
            and isinstance(content, str)
            and "es correcto" in content.lower()
        ):
            confirmacion_idx = idx
            logging.info(f"Confirmación encontrada en mensaje [{idx}]")
            break

    if confirmacion_idx is None:
        logging.info("No se encontró mensaje de confirmación.")
        return None

    logging.info("Buscando mensaje de pedido anterior a la confirmación...")

    # Buscar hacia atrás un mensaje enviado que parezca un pedido
    for idx in range(confirmacion_idx - 1, -1, -1):
        direction = messages[idx].direction
        content = messages[idx].content
        if direction == "sent" and isinstance(content, str):
            logging.info(f"Revisando posible pedido en mensaje [{idx}]")

            content_stripped = content.strip().lower()
            if content_stripped.startswith("pedido:"):
                # Verifica tokens del tipo "\codigo"
                tokens = re.findall(r"\\\S+", content)
                logging.info(
                    f"Mensaje [{idx}] contiene {len(tokens)} tokens con formato '\\...': {tokens}"
                )
                if len(tokens) >= 4:
                    pedido_idx = idx
                    logging.info(f"Pedido válido encontrado en mensaje [{idx}]")
                    break
                else:
                    logging.info(
                        f"Mensaje [{idx}] comienza con 'pedido:' pero no contiene suficientes tokens válidos."
                    )

    if pedido_idx is None:
        logging.info("No se encontró ningún mensaje de pedido válido.")
        return None

    logging.info(f"Retornando contenido del pedido del mensaje [{pedido_idx}]")
    return messages[pedido_idx].content


def order_to_xlsx(ocr_text: str) -> Optional[str]:
    """
    Convierte un texto OCR plano en un archivo Excel (.xlsx) con columnas Código y Cantidad.
    El texto debe tener el formato: "PEDIDO: \codigo \cantidad \codigo \cantidad ..."
    """
    try:
        parts = ocr_text.strip().split("PEDIDO:")
        if len(parts) != 2:
            return None
        payload = parts[1].strip()
    except Exception:
        return None

    tokens = [t.strip() for t in payload.split("\\") if t.strip()]

    if len(tokens) % 2 != 0:
        return None  # Esperamos pares (codigo, cantidad)

    data = [(tokens[i], tokens[i + 1]) for i in range(0, len(tokens), 2)]

    if not data:
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        path = tmp.name

    wb = Workbook()
    ws = wb.active
    ws.title = "Pedido"
    ws.append(["CodigoArticulo", "Unidades"])
    for row in data:
        ws.append(row)

    wb.save(path)
    return path


def order_to_pdf(ocr_text: str, output_path: Optional[str] = None) -> str:
    """
    Convierte el texto OCR plano del pedido en un PDF.
    Formato esperado: "PEDIDO: \codigo \cantidad \codigo \cantidad ..."
    """
    try:
        parts = ocr_text.strip().split("PEDIDO:")
        if len(parts) != 2:
            raise ValueError("No se encontró 'PEDIDO:' en el OCR")

        payload = parts[1].strip()
    except Exception:
        raise ValueError("Formato OCR inválido")

    tokens = [t.strip() for t in payload.split("\\") if t.strip()]

    if len(tokens) % 2 != 0:
        raise ValueError("Número impar de valores en bloque de pedido")

    data = [["Código", "Cantidad"]]
    for i in range(0, len(tokens), 2):
        data.append([tokens[i], tokens[i + 1]])

    if not output_path:
        output_path = os.path.join(tempfile.gettempdir(), "pedido.pdf")

    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("Pedido de productos", styles["Title"]),
        Spacer(1, 12),
        Table(
            data,
            colWidths=[200, 100],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            ),
        ),
    ]

    doc.build(elements)
    return output_path
