# Slim Python base
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (use layer cache)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project (expects api/, scripts/, docs/, vector_store/, .env in build context)
COPY api /app/api
COPY scripts /app/scripts
COPY docs /app/docs
COPY vector_store /app/vector_store
COPY .env /app/.env
COPY web /app/web

# Runtime env
ENV PERSIST_DIR=/app/vector_store \
    HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Healthcheck FastAPI
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Start API
CMD ["python","-m","uvicorn","api.app:app","--host","0.0.0.0","--port","8000"]
