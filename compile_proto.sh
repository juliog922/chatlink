#!/bin/bash

# Path to the proto file
PROTO_PATH="proto/whatsapp.proto"

echo "Compiling proto for Python Client..."

# Ensure target dir exists
mkdir -p chatlink_bot/src/chatlink_bot

# Generate Python code
# We output directly into the package directory
python -m grpc_tools.protoc \
  -Iproto \
  --python_out=chatlink_bot/src/chatlink_bot \
  --grpc_python_out=chatlink_bot/src/chatlink_bot \
  $PROTO_PATH

# Fix import issue in generated grpc file (Python 3 relative import fix)
# This replaces "import whatsapp_pb2" with "from . import whatsapp_pb2"
sed -i 's/import whatsapp_pb2 as whatsapp__pb2/from . import whatsapp_pb2 as whatsapp__pb2/' chatlink_bot/src/chatlink_bot/whatsapp_pb2_grpc.py

echo "✅ Python compilation completed."