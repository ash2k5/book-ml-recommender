FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH"

# Install runtime dependencies from the lockfile (cached unless deps change)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Application code and committed catalog
COPY bookrec ./bookrec
COPY data/books.csv ./data/books.csv

EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn bookrec.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
