import io
import logging
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import paramiko
import os
import textwrap
from dotenv import load_dotenv

load_dotenv()

# SFTP config (cargar antes desde dotenv)
SFTP_HOST: str = os.getenv("SFTP_HOST", "")
SFTP_PORT: int = int(os.getenv("SFTP_PORT", "22"))
SFTP_USERNAME: str = os.getenv("SFTP_USERNAME", "")
SFTP_PASSWORD: str = os.getenv("SFTP_PASSWORD", "")
SFTP_REMOTE_DIR: str = os.getenv("SFTP_REMOTE_DIR", "articulos")


def connect_sftp() -> Tuple[paramiko.SFTPClient, paramiko.Transport]:
    transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
    transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)
    return sftp, transport


def find_image_file(image_name: str) -> Optional[bytes]:
    """
    Finds and returns the image content from SFTP (as bytes), or None if not found.
    """
    sftp, transport = connect_sftp()
    try:
        sftp.chdir(SFTP_REMOTE_DIR)
        files = sftp.listdir()
        target_filename = f"{image_name}.jpg"

        matches = [
            f
            for f in files
            if f == target_filename
            and f.lower().endswith(".jpg")
            and "mini" not in f.lower()
            and "_" not in f
        ]

        if not matches:
            return None

        with sftp.open(matches[0], "rb") as f:
            return f.read()
    except Exception as e:
        logging.warning(f"Error loading image {image_name}: {e}")
        return None
    finally:
        sftp.close()
        transport.close()


def build_order_image_table(
    items: List[Tuple[str, str, str, Optional[bytes]]],
    font_path: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    font_size: int = 18,
    cell_padding: int = 10,
    thumb_size: Tuple[int, int] = (100, 100),
) -> Image.Image:

    gray_text_color = "#CCCCCC"
    black_text_color = "#000000"

    headers = ["Código", "Cantidad", "Descripción", "Imagen"]
    col_widths = [180, 140, 400, thumb_size[0] + 2 * cell_padding]
    base_row_height = max(
        thumb_size[1] + 2 * cell_padding, font_size + 2 * cell_padding
    )

    def get_font(size: int) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(font_path, size)

    base_font = get_font(font_size)
    ocr_font = get_font(font_size + 4)

    # Estimar alturas de fila
    row_heights = []
    for _, _, descripcion, _ in items:
        max_lines = 4
        current_font_size = font_size
        while current_font_size >= 10:
            font = get_font(current_font_size)
            line_height = font.getbbox("A")[3] - font.getbbox("A")[1]
            wrapped = textwrap.wrap(descripcion, width=26)
            if len(wrapped) <= max_lines:
                break
            current_font_size -= 1
        row_heights.append(
            max(base_row_height, current_font_size * max_lines + 2 * cell_padding)
        )

    header_height = font_size * 3
    total_height = sum(row_heights) + base_row_height + header_height + font_size + 10
    total_width = sum(col_widths)

    image = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(image)

    y = 0

    # Título
    draw.text((cell_padding, y), "PEDIDO:", font=ocr_font, fill=black_text_color)
    y += font_size + 10

    # Encabezados
    x = 0
    for i, header in enumerate(headers):
        draw.rectangle(
            [x, y, x + col_widths[i], y + base_row_height],
            fill="#EEEEEE",
            outline="gray",
        )
        draw.text(
            (x + cell_padding, y + cell_padding),
            header,
            font=base_font,
            fill=gray_text_color,
        )
        x += col_widths[i]
    y += base_row_height

    # Filas
    for idx, (codigo, cantidad, descripcion, img_bytes) in enumerate(items):
        row_height = row_heights[idx]
        x = 0

        # Código
        draw.rectangle([x, y, x + col_widths[0], y + row_height], outline="gray")
        draw.text(
            (x + cell_padding, y + cell_padding),
            codigo,
            font=ocr_font,
            fill=black_text_color,
        )
        x += col_widths[0]

        # Cantidad
        draw.rectangle([x, y, x + col_widths[1], y + row_height], outline="gray")
        draw.text(
            (x + cell_padding, y + cell_padding),
            cantidad,
            font=ocr_font,
            fill=black_text_color,
        )
        x += col_widths[1]

        # Descripción (ajustada a 26 caracteres por línea)
        draw.rectangle([x, y, x + col_widths[2], y + row_height], outline="gray")
        max_width = col_widths[2] - 2 * cell_padding
        max_height = row_height - 2 * cell_padding
        current_font_size = font_size

        while current_font_size >= 10:
            font = get_font(current_font_size)
            line_height = font.getbbox("A")[3] - font.getbbox("A")[1]
            wrapped = textwrap.wrap(descripcion, width=26)
            total_height = len(wrapped) * line_height
            if total_height <= max_height:
                break
            current_font_size -= 1

        font = get_font(current_font_size)
        line_height = font.getbbox("A")[3] - font.getbbox("A")[1]
        max_lines = max_height // line_height
        wrapped = textwrap.wrap(descripcion, width=26)[:max_lines]

        if len(textwrap.wrap(descripcion, width=26)) > max_lines:
            wrapped[-1] = wrapped[-1][: max(0, len(wrapped[-1]) - 1)] + "\\"

        desc_y = y + cell_padding
        line_spacing = int(
            line_height * 0.2
        )  # Agrega 20% de espacio adicional entre líneas
        for line in wrapped:
            draw.text((x + cell_padding, desc_y), line, font=font, fill=gray_text_color)
            desc_y += line_height + line_spacing
        x += col_widths[2]

        # Imagen o texto alternativo
        draw.rectangle([x, y, x + col_widths[3], y + row_height], outline="gray")
        if img_bytes:
            try:
                thumb = Image.open(io.BytesIO(img_bytes))
                thumb.thumbnail(thumb_size)
                image.paste(thumb, (x + cell_padding, y + cell_padding))
            except Exception:
                draw.text(
                    (x + cell_padding, y + cell_padding),
                    "Error imagen",
                    font=base_font,
                    fill=gray_text_color,
                )
        else:
            msg_font = get_font(font_size - 2)
            msg = "Imagen no\ndisponible"
            draw.multiline_text(
                (x + cell_padding, y + cell_padding),
                msg,
                font=msg_font,
                fill=gray_text_color,
            )

        y += row_height

    return image
