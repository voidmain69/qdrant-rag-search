# --- build stage: resolve and install dependencies with uv ---
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY app ./app
COPY scripts ./scripts
COPY data ./data

# --- runtime stage ---
FROM python:3.13-slim-bookworm

RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /models && chown appuser:appuser /models

WORKDIR /app
COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/models \
    HF_HOME=/models/hf

USER appuser
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
