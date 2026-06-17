# Changelog

## 2026-06-14

- Replaced the Flask + Jinja web layer with a FastAPI JSON API (`bookrec/api.py`): endpoints
  for books, recommendations, related, genres, search, and health, plus an OpenAPI schema. The
  TF-IDF recommender and dataset are unchanged.
- Bounded and validated query params (`limit`, `n`); unknown book IDs return 404.
- Runs on uvicorn; `python -m bookrec` starts the dev server on 8000.
- Added CORS via `ALLOWED_ORIGINS`.
- Dropped the server-rendered templates and the gunicorn/WSGI entry point.

## 2026-06-12

- Rebuilt search results with DOM APIs and moved book-card navigation to a delegated
  `data-book-id` listener.
- Moved to pandas 3, numpy 2, and scikit-learn 1.9, each pinned to a tested major range.
- Added a pip-audit dependency scan in CI.
- Removed the stale `notebooks/` analysis; evaluation lives in `scripts/evaluate.py`.

## 2026-06-06

- `bookrec` package: a validated CSV loader, an in-memory TF-IDF recommender, and a
  `create_app` factory.
- A ~1,500-book catalog (a subset of Best Books Ever) with build and evaluation scripts,
  packaging, a lockfile, and CI.
- Book detail pages read IDs as strings so URL lookups match; search treats the query
  literally; recommendations are built in memory at startup.
