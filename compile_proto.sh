#!/bin/bash

# Ruta base
PROTO_PATH="proto/whatsapp.proto"

# === Go ===
echo "Compiling proto on Go Server..."
protoc \
  --go_out=whatsmeow_go/src \
  --go-grpc_out=whatsmeow_go/src \
  $PROTO_PATH

# === Python ===
echo "Compiling proto on Python Client..."
python -m grpc_tools.protoc \
  -Iproto \
  --python_out=whatsapp_bot/src/proto \
  --grpc_python_out=whatsapp_bot/src/proto \
  $PROTO_PATH

echo "✅ Compilación completada."
