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
COPY templates ./templates
COPY static ./static
COPY data/books.csv ./data/books.csv
COPY wsgi.py ./

EXPOSE 8000
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 4 --timeout 120 wsgi:app"]
