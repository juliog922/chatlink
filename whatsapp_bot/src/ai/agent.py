import logging
import tempfile
import time
from typing import List, Optional
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage
from sqlalchemy import and_, func
from sqlalchemy.orm import Session
from PIL.Image import Image
from sqlalchemy.orm import aliased
from sqlalchemy import select, desc
from dotenv import load_dotenv
import os
from datetime import timedelta, datetime

from src.ai.extractors import (
    extract_response_text,
    extract_mentioned_products,
    is_order,
    is_order_confirmation
)
from src.ai.utils import update_order, confirmed_order, order_to_xlsx, order_to_pdf
from src.core.database import get_sqlite_session, get_sqlserver_session
from src.models.message import Message
from src.models.user import User
from src.models.product import Articulo
from src.models.client import Cliente
from src.grpc.handlers import send_message, send_file
from src.mail.mail_handler import notify_order_by_email

load_dotenv()
MIN_MINUTES = int(os.getenv("UNATTENDED_MINUTES_MIN", 15))
MAX_MINUTES = int(os.getenv("UNATTENDED_MINUTES_MAX", 30))

OLLAMA_URL = os.getenv("OLLAMA_URL", "")

chat = ChatOllama(
    model="llama3",
    temperature=0.0,
    top_p=0.1,
    repeat_penalty=1.2,
    num_predict=128,
    format="json",
)


def mentioned_products_prompt(history: str, message_text: str) -> str:
    return f"""
    <|begin_of_text|>
    <|system|>
    Eres una IA que asiste a un comercial de cosméticos. Tu única tarea es analizar pedidos escritos por clientes para extraer productos que cumplan lo siguiente:

    1. Incluyan un **código de producto válido** (ej: "KG990A", "00123", "A100").
    2. Opciónalmente, incluyan una **cantidad** asociada (ej: “x3”, “dos”, “una más”, “4 unidades”).

    Tu respuesta debe ser **EXCLUSIVAMENTE un JSON válido**, con este formato:

    {{
    "items": [
        ["<código>", "<cantidad>"],
        ...
    ]
    }}

    ### Instrucciones detalladas:
    - Extrae productos **solo si** tienen un código claro y explícito (alfanumérico, sin ambigüedad).
    - Ignora cualquier otro detalle que no sean **codigo y cantidad**.
    - La **cantidad** debe expresarse como número entero (ej: “uno” → "1", “x3” → "3").
    - Si **no se menciona cantidad**, **no incluyas ese código**.
    - Si un código aparece varias veces con cantidades, **suma las cantidades**.
    - Si se menciona eliminar un producto, **no lo incluyas**.
    - Si se indica una corrección, **reemplaza la cantidad anterior**.
    - Si se dice “mejor”, “cambia”, “en lugar de”, “corrige”, **usa solo la cantidad más reciente** para ese código.
    - Ignora referencias vagas como “ese”, “el anterior”, “el otro”.
    - Si hay mezcla de frases sociales y códigos, **solo extrae los códigos**.
    - Si **no hay códigos válidos**, responde: `{{ "items": [] }}`

    También debes considerar el **historial de mensajes anteriores** para detectar si el cliente está **corrigiendo** un pedido previo.

    ### Ejemplos:

    <|user|>
    Historial:
    - Cliente: PEDIDO: \\8741 \\1 \\GFT543 \\3 \\7787548 \\25 \\HGT6554 \\1
    Mensaje: Corrige, ponme 5 del FFFFF y 2 más del 8741

    <|assistant|>
    {{
    "items": [
        ["8741", "3"],
        ["GFT543", "3"],
        ["7787548", "25"],
        ["HGT6554", "1"],
        ["FFFFF", "5"]
    ]
    }}

    <|user|>
    Historial:
    - Cliente: Pasame dos del X8876287
    - Comercial: Listo, anotado
    Mensaje: Ah, mejor ponme cuatro del X8876287

    <|assistant|>
    {{
    "items": [
        ["X8876287", "4"]
    ]
    }}

    <|user|>
    Historial:
    {history}

    Mensaje:
    {message_text}
    <|assistant|>
    """.strip()


def is_order_prompt(history: str, message_text: str) -> str:
    return f"""
    <|begin_of_text|>
    <|system|>
    Eres una IA que asiste a un comercial de cosméticos. Tu única tarea es decidir si el cliente está haciendo un pedido.

    Responde SOLO con uno de estos JSON válidos:
    - Si es un pedido: {{ "order": true }}
    - Si no lo es: {{ "order": false }}

    Cuenta como pedido:
    - Debe mencionar codigos junto con cantidades.
    - Correccion a un pedido anterior igualmente indicando codigo y cantidad.
    - Mencionar productos con códigos alfanuméricos (ej: KG990A, A100)
    - Usar frases como "pasame", "poneme", "sumale", "agregá", "mandame", "quiero", etc.

    No cuenta como pedido:
    - Pedidos ya confirmados.
    - Consultas sin códigos (¿Tenés algo nuevo?)
    - Confirmaciones ("es correcto", "gracias")
    - Dudas o frases vagas ("después te paso", "estoy viendo")
    - Solo la intencion de hacerlo sin describir el pedido.

    Ejemplos: 
    
    <|user|>
    Historial:
    - Cliente: Código del nuevo labial?
    - Comercial: 998ZT
    Mensaje: Pasame 2 del 998ZT y 3 del A100

    <|assistant|>
    {{ "order": true }}

    <|user|>
    Historial:
    - Cliente: Me encantó el pedido anterior
    - Comercial: Qué bueno
    Mensaje: Capaz más adelante pida algo

    <|assistant|>
    {{ "order": false }}

    <|user|>
    Historial:
    - Comercial: Pedido cargado
    Mensaje: Es correcto

    <|assistant|>
    {{ "order": false }}

    <|user|>
    Historial:
    {history}

    Mensaje:
    {message_text}
    <|assistant|>
    """.strip()


def chat_prompt(comercial_name: str, history: str, message_text: str) -> str:
    return f"""
    <|start_header_id|>system<|end_header_id|>
    Eres el asistente virtual de Kapalua, distribuidor de cosmética. Atiendes solo si el mensaje tiene **una intención claramente comercial**. Tu objetivo es ayudar a guiar al cliente para la creación de su pedido. Eres el asistente de creacion de pedidos en Whatsapp de {comercial_name}.

    No debes responder si:
    - Es un saludo, despedida, emoji o comentario sin fin comercial.
    - El cliente dice que hablará o esperará al comercial.
    - Ya está en conversación con el comercial.
    - El mensaje es ambiguo o no comercial.

    Si el mensaje **sí es comercial**, responde solo en dos casos:
    1. **Si el cliente pregunta cómo hacer un pedido** → Explícale que debe enviar cantidades junto con los códigos de producto. Ejemplo: `2 x X8876287`, `3 x KG500`. Puede hacerlo por texto, audio, imagen clara (no manuscrita) o archivo (PDF, CSV o TXT).
    2. **Para cualquier otra consulta comercial** (productos, promociones, incidencias, **precios**, detalles de productos, etc.) → Di que el comercial {comercial_name} lo atenderá lo antes posible. No debes dar detalles de estos temas.

    Cuando el cliente envie una propuesta de pedido, se le enviara un mensaje de confirmacion mostrando su pedido agregando imagenes de productos y nombre oficial para que el cliente pueda confirmar o corregir.
    Cuando sea necesario recuerdale al cliente que el pedido se compone unicamente de codigos mas cantidades para evitar errores e inconvenientes.
    En caso de que el cliente tenga problemas con tu ayuda o si lo ves conveniente recuerda al cliente que puede dejar de responderte para esperar que {comercial_name} solucione sus dudas, incovenientes o gestion del pedido.

    Sé profesional, claro y directo. No respondas nada fuera del ámbito comercial.

    Tu salida debe ser siempre un JSON válido.

    Si **NO debes responder**:
    {{ "responder": false }}

    Si **SÍ debes responder**:
    {{ 
    "responder": true,
    "respuesta": "..." 
    }}
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Historial:
    {history}

    Mensaje del cliente:
    {message_text}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """.strip()


def handle_incoming_message(
    sqlite_session: Session,
    sqlserver_session: Session,
    stub,
    receiver: str,
    sender: str,
    message_text: str,
    chat=chat,
):
    logging.info("Handling incoming message for AI processing")

    cliente = Cliente.get_by_telefono(sqlserver_session, sender)
    if not cliente:
        logging.warning(f"There is not a client with phone: {sender}")
        return

    comercial = User.get_by_phone(sqlite_session, receiver)
    if not comercial:
        logging.warning(f"There is not user with phone: {receiver}")
        return

    comercial_name: str = comercial.name or "el vendedor"

    stmt = (
        select(Message.direction, Message.content)
        .where(Message.client_id == cliente.codigo_cliente)
        .order_by(desc(Message.timestamp))
        .limit(6)
    )
    messages: List[Message] = sqlite_session.execute(stmt).all()[::-1]

    history: str = "\n".join(
        [
            f"{'Cliente:' if d == 'sent' else 'Comercial:'}: {c}"
            for d, c in messages
            if c
        ]
    )

    is_order_prompt_text: str = is_order_prompt(history, message_text)
    is_order_raw_response: str = chat.invoke(
        [HumanMessage(content=is_order_prompt_text)]
    ).content.strip()
    logging.info(f"Is an order: {is_order(is_order_raw_response)}")
    if is_order(is_order_raw_response):
        logging.info(f"Is an order confirmation: {is_order_confirmation(message_text)}")
        if is_order_confirmation(message_text):
            for message in messages:
                logging.info(
                    f"message direction: {message.direction} \ message content: {message.content}"
                )
            confirmed_order_text: str = confirmed_order(messages)
            logging.info(f"confirmed_order_text: {confirmed_order_text}")
            updated_confirmed_order_csv_path: Optional[str] = order_to_xlsx(
                confirmed_order_text
            )
            updated_confirmed_order_pdf_path: str = order_to_pdf(confirmed_order_text)

            notify_order_by_email(
                user=comercial,
                client=cliente,
                phone=sender,
                csv_path=updated_confirmed_order_csv_path,
            )
            send_file(stub, sender, updated_confirmed_order_pdf_path, from_jid=receiver)
        else:
            mentioned_products_prompt_text: str = mentioned_products_prompt(
                history, message_text
            )
            mentioned_products_raw_response: str = chat.invoke(
                [HumanMessage(content=mentioned_products_prompt_text)]
            ).content.strip()
            if mentioned_products := extract_mentioned_products(
                mentioned_products_raw_response
            ):
                logging.info(f"Mentioned products: {mentioned_products}")
                img: Image | None = update_order(sqlserver_session, mentioned_products)
                if img:
                    send_message(
                        stub,
                        sender,
                        "Confirma si el pedido es correcto respondiendo con *Es correcto*.\
                        Se lo pasaremos a tu comercial que se encargará de todo o te contactará si hay alguna duda.\
                        En caso de que no sea correcto, siente libre de repetirme el pedido o indicar unicamente las correcciones\
                        [Este mensaje fue generado automáticamente por un asistente en versión de pruebas]",
                        from_jid=receiver,
                    )
                    
                    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
                    filename = f"pedido_{timestamp}.jpg"
                    tmp_dir = tempfile.gettempdir()
                    filepath = os.path.join(tmp_dir, filename)

                    img.save(filepath, format="JPEG")

                    send_file(stub, sender, filepath=filepath, from_jid=receiver)
                    os.remove(filepath)
            else:
                chat_prompt_text: str = chat_prompt(
                    comercial_name, history, message_text
                )
                chat_raw_response: str = chat.invoke(
                    [HumanMessage(content=chat_prompt_text)]
                ).content.strip()
                chat_response: str | None = extract_response_text(chat_raw_response)
                if chat_response and len(chat_response.strip()) > 0:
                    chat_response += "\n[Este mensaje fue generado automáticamente por un asistente en versión de pruebas]"
                    send_message(stub, sender, chat_response, from_jid=receiver)
                    logging.info("IA Response successfully sent")
                else:
                    logging.info("There is not IA response")
    else:
        chat_prompt_text: str = chat_prompt(comercial_name, history, message_text)
        chat_raw_response: str = chat.invoke(
            [HumanMessage(content=chat_prompt_text)]
        ).content.strip()
        chat_response: str | None = extract_response_text(chat_raw_response)
        if chat_response and len(chat_response.strip()) > 0:
            chat_response += "\n[Este mensaje fue generado automáticamente por un asistente en versión de pruebas]"
            send_message(stub, sender, chat_response, from_jid=receiver)
            logging.info("IA Response successfully sent")
        else:
            logging.info("There is not IA response")


def search_products(sqlserver_session: Session, keywords: list[str]) -> str:
    if not keywords:
        return ""

    articulos = Articulo.get_by_words_list(sqlserver_session, keywords)

    if not articulos:
        return ""

    lines = []
    for art in articulos:
        lines.append(
            f"- Codigo de Articulo: {art.codigo} / Descripcion: {art.descripcion1}"
        )

    logging.info("\n".join(lines))
    return "\n".join(lines)


def process_unattended_messages_loop(stub):
    while True:
        logging.info("🔍 Revisando últimos mensajes de clientes no respondidos...")

        sqlite_session = get_sqlite_session()
        sqlserver_session = get_sqlserver_session()

        try:
            # Aliased para evitar conflictos
            MessageAlias = aliased(Message)

            # Subconsulta que obtiene el último timestamp por client_id
            subq = (
                sqlite_session.query(
                    MessageAlias.client_id,
                    func.max(MessageAlias.timestamp).label("latest"),
                )
                .filter(MessageAlias.direction == "received")
                .group_by(MessageAlias.client_id)
                .subquery()
            )

            # Join para obtener los mensajes completos
            last_msgs = (
                sqlite_session.query(MessageAlias)
                .join(
                    subq,
                    and_(
                        MessageAlias.client_id == subq.c.client_id,
                        MessageAlias.timestamp == subq.c.latest,
                    ),
                )
                .filter(MessageAlias.direction == "received")
                .all()
            )
            now = datetime.now()

            for last_msg in last_msgs:
                # Verificar si ya se respondió
                response = (
                    sqlite_session.query(Message)
                    .filter(
                        and_(
                            Message.client_id == last_msg.client_id,
                            Message.direction == "sent",
                            Message.timestamp > last_msg.timestamp,
                        )
                    )
                    .first()
                )

                if response:
                    continue  # Ya respondido

                if not last_msg.content:
                    continue  # No hay texto para procesar

                # Verificar rango de tiempo
                age = now - last_msg.timestamp
                if age < timedelta(minutes=MIN_MINUTES) or age > timedelta(
                    minutes=MAX_MINUTES
                ):
                    continue  # Fuera del rango definido

                # Obtener teléfono del cliente
                client_phone = last_msg.client_phone
                client = Cliente.get_by_telefono(sqlserver_session, client_phone)
                if not client:
                    continue

                # Obtener usuario asignado
                user = (
                    sqlite_session.query(User)
                    .filter(User.id == last_msg.user_id)
                    .first()
                )
                if not user:
                    logging.info(f"Cliente {last_msg.client_id} sin usuario asignado")
                    continue

                logging.info(f"🤖 Enviando respuesta IA a cliente {last_msg.client_id}")
                handle_incoming_message(
                    sqlite_session,
                    sqlserver_session,
                    stub,
                    user.phone,
                    client_phone,
                    last_msg.content,
                )

        except Exception as e:
            logging.error(f"Error en el loop de mensajes no atendidos: {e}")

        finally:
            sqlite_session.close()
            sqlserver_session.close()

        time.sleep(60)


def process_one_unattended_batch(sqlite_session, sqlserver_session, stub, send_fn):
    subquery = (
        sqlite_session.query(
            Message.client_id, func.max(Message.timestamp).label("latest")
        )
        .filter(Message.direction == "received")
        .group_by(Message.client_id)
        .subquery()
    )

    last_msgs = (
        sqlite_session.query(Message)
        .join(
            subquery,
            and_(
                Message.client_id == subquery.c.client_id,
                Message.timestamp == subquery.c.latest,
            ),
        )
        .all()
    )

    now = datetime.now()

    for last_msg in last_msgs:
        # Rango de tiempo
        age = now - last_msg.timestamp
        if age < timedelta(minutes=MIN_MINUTES) or age > timedelta(minutes=MAX_MINUTES):
            continue

        response = (
            sqlite_session.query(Message)
            .filter(
                and_(
                    Message.client_id == last_msg.client_id,
                    Message.direction == "sended",
                    Message.timestamp > last_msg.timestamp,
                )
            )
            .first()
        )

        if response or not last_msg.content:
            continue

        telefono = last_msg.sender
        if not telefono:
            continue

        cliente = Cliente.get_by_telefono(sqlserver_session, telefono)
        if not cliente:
            continue

        user = sqlite_session.query(User).filter(User.id == last_msg.user_id).first()
        if not user:
            continue

        send_fn(
            sqlite_session,
            sqlserver_session,
            stub,
            send_message,
            user.phone,
            telefono,
            last_msg.content,
        )


def search_simulated_products(fake_index: dict[str, str], keywords: list[str]) -> str:
    if not keywords:
        return ""

    results = []
    for key in keywords:
        for code_or_kw, desc in fake_index.items():
            if key.lower() in code_or_kw.lower() or key.lower() in desc.lower():
                results.append(
                    f"- Codigo de Articulo: {code_or_kw} / Descripcion: {desc}"
                )
    return "\n".join(results)
