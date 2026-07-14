# -------- Build Stage --------
FROM golang:latest AS builder

WORKDIR /src

# Set environment for static build
ENV CGO_ENABLED=0 \
    GOOS=linux \
    GOPROXY=https://proxy.golang.org,direct \
    GOSUMDB=on

# 1. Download grpc-health-probe binary
# Added "mkdir -p /out" to ensure the directory exists
RUN GRPC_HEALTH_PROBE_VERSION=v0.4.28 && \
    apt-get update && apt-get install -y curl && \
    mkdir -p /out && \
    curl -L https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64 -o /out/grpc_health_probe && \
    chmod +x /out/grpc_health_probe

# 2. Copy Source Code
COPY . .

# 3. Initialize Module (Robustness)
RUN [ -f go.mod ] || go mod init github.com/juliog922/meow_server || true

# 4. Fetch Dependencies & Build
RUN go mod tidy && \
    go build -trimpath -ldflags="-s -w -buildid=" -o /out/wa-grpc ./

# -------- Final Stage --------
FROM gcr.io/distroless/static-debian12:nonroot

WORKDIR /app

# Copy the app binary
COPY --from=builder /out/wa-grpc /app/wa-grpc

# Copy the health probe binary
COPY --from=builder /out/grpc_health_probe /bin/grpc_health_probe

# Default environment variables
ENV WHATSMEOW_HOST=0.0.0.0 \
    WHATSMEOW_PORT=50051

EXPOSE 50051

USER nonroot:nonroot

ENTRYPOINT ["/app/wa-grpc"]