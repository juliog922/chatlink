# Switched to official Python Slim image (Debian Bookworm)
# Reason: Massive size reduction since CUDA/GPU is no longer required.
FROM python:3.12-slim-bookworm

# Environment Optimizations
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    DEBIAN_FRONTEND=noninteractive \
    ACCEPT_EULA=Y

WORKDIR /app

# 1. Install System Dependencies
# 'unixodbc-dev': Required for compiling/linking ODBC drivers (referenced in database.py).
RUN apt-get update && apt-get install -y --no-install-recommends \
    unixodbc \
    unixodbc-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2. Install ODBC Driver 17 (Local)
# Reason: Installing from local .deb to avoid external network dependencies.
COPY msodbcsql17_17.10.6.1-1_amd64.deb /tmp/msodbcsql17.deb

RUN apt-get update && apt-get install -y --no-install-recommends \
    /tmp/msodbcsql17.deb \
    && rm /tmp/msodbcsql17.deb \
    && rm -rf /var/lib/apt/lists/*

# 3. Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 4. Copy Config Files
COPY pyproject.toml README.md ./

# 5. Create Venv & Install Dependencies
RUN uv venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# Install deps (ensuring no-cache to keep image small)
RUN uv sync --no-install-project --no-cache

# 6. Copy Source Code
COPY src/ ./src/

# 7. Final Setup
EXPOSE 8000

ENV PYTHONPATH="/app/src"

CMD ["uvicorn", "chatlink_bot.main:app", "--host", "0.0.0.0", "--port", "8000"]