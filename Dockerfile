# Stage 1: resolve locked serving dependencies to a requirements file
FROM python:3.11-slim AS builder
RUN pip install --no-cache-dir poetry==2.4.1 poetry-plugin-export
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry export --only main -f requirements.txt -o requirements.txt

# Stage 2: runtime image
FROM python:3.11-slim
# libgomp1: LightGBM runtime dependency missing from slim images
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY ml/ ./ml/
# Numeric UID so Kubernetes runAsNonRoot can verify the user is non-root
RUN useradd -m -u 10001 appuser && chown -R appuser /app
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

ENV UVICORN_RELOAD=""
CMD ["sh", "-c", "uvicorn app.main:app --host=0.0.0.0 --port=8000 ${UVICORN_RELOAD}"]
