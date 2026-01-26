# -------- Build Stage --------
    FROM golang:1.24-bookworm AS builder

    WORKDIR /src
    
    # Set environment for static build
    # GOPROXY=https://goproxy.io,direct: Use a reliable public proxy to avoid connection resets
    # GOSUMDB=off: Disable checksum validation to prevent further network blockers
    ENV CGO_ENABLED=0 \
        GOOS=linux \
        GOPROXY=https://goproxy.io,direct \
        GOSUMDB=off
    
    # 1. Install grpc-health-probe
    RUN go install github.com/grpc-ecosystem/grpc-health-probe@latest
    
    # 2. Copy Source Code
    COPY . .
    
    # 3. Initialize Module (Robustness)
    RUN go mod init github.com/juliog922/meow_server || true
    
    # 4. Force Update Dependencies & Build
    # We use the proxy now, so this should succeed without network errors
    RUN go get go.mau.fi/whatsmeow@latest && \
        go get github.com/lib/pq@latest && \
        go mod tidy && \
        go build -trimpath -ldflags="-s -w -buildid=" -o /out/wa-grpc ./
    
    # -------- Final Stage --------
    FROM gcr.io/distroless/static-debian12:nonroot
    
    WORKDIR /app
    
    # Copy the app binary
    COPY --from=builder /out/wa-grpc /app/wa-grpc
    
    # Copy the health probe binary
    COPY --from=builder /go/bin/grpc-health-probe /bin/grpc_health_probe
    
    # Default environment variables
    ENV WHATSMEOW_HOST=0.0.0.0 \
        WHATSMEOW_PORT=50051
    
    EXPOSE 50051
    
    USER nonroot:nonroot
    
    ENTRYPOINT ["/app/wa-grpc"]