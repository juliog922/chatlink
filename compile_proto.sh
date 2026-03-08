#!/bin/bash

# Path to the proto file
PROTO_PATH="proto/whatsapp.proto"

echo "========================================"
echo "🐍 Compiling proto for Python Client..."
echo "========================================"

# Ensure target dir exists
mkdir -p chatlink_bot/src/chatlink_bot

# Generate Python code
# We output directly into the package directory
python3 -m grpc_tools.protoc \
  -Iproto \
  --python_out=chatlink_bot/src/chatlink_bot \
  --grpc_python_out=chatlink_bot/src/chatlink_bot \
  $PROTO_PATH

sed -i 's/^import whatsapp_pb2 as whatsapp__pb2/from . import whatsapp_pb2 as whatsapp__pb2/' chatlink_bot/src/chatlink_bot/whatsapp_pb2_grpc.py

echo "✅ Python compilation completed."
echo ""

echo "========================================"
echo "🐹 Compiling proto for Go Server..."
echo "========================================"

# Ensure target dir exists
# Because the proto file has 'option go_package = "./proto";', 
# passing meow_server/src as the out dir will correctly place 
# the files in meow_server/src/proto/
mkdir -p meow_server/src

# Generate Go code
protoc \
  -Iproto \
  --go_out=meow_server/src \
  --go-grpc_out=meow_server/src \
  $PROTO_PATH

echo "✅ Go compilation completed."