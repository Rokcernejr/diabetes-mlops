FROM python:3.11

WORKDIR /app

# Install core packages first (including prometheus-client)
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    "uvicorn[standard]>=0.24.0" \
    pydantic>=2.5.0 \
    prometheus-client>=0.19.0

# Copy pyproject.toml and install via Poetry (for dependency management)
COPY pyproject.toml ./
RUN pip install poetry==1.8.2 && \
    poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi

# Copy application code
COPY app/ ./app/
COPY ml/ ./ml/
RUN touch app/__init__.py ml/__init__.py

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENV UVICORN_RELOAD=""
CMD ["sh", "-c", "uvicorn app.main:app --host=0.0.0.0 --port=8000 ${UVICORN_RELOAD}"]