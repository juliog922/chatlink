import grpc
import logging
import time

from src.proto.whatsapp_pb2_grpc import WhatsAppServiceStub


def create_grpc_stub(host="localhost", port=50051) -> WhatsAppServiceStub:
    address = f"{host}:{port}"
    while True:
        try:
            channel = grpc.insecure_channel(
                address,
                options=[
                    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                    ("grpc.max_send_message_length", 64 * 1024 * 1024),
                ],
            )
            grpc.channel_ready_future(channel).result(timeout=5)
            logging.info(f"Connected to gRPC server at {address}")
            return WhatsAppServiceStub(channel)
        except Exception as e:
            logging.warning(f"Waiting for gRPC server at {address}... ({e})")
            time.sleep(2)
