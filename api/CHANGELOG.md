# changelog

## 2026-06-14

- replaced the flask + jinja web layer with a fastapi json api (`bookrec/api.py`):
  endpoints for books, recommendations, related, genres, search, and health, plus an
  openapi schema. the tf-idf recommender and dataset are unchanged.
- bounded and validated query params (`limit`, `n`); unknown book ids return 404.
- runs on uvicorn; `python -m bookrec` starts the dev server on 8000.
- added cors via `ALLOWED_ORIGINS`.
- dropped the server-rendered templates and the gunicorn/wsgi entry point.

## 2026-06-12

- rebuilt search results with dom apis and moved book-card navigation to a delegated
  `data-book-id` listener.
- moved to pandas 3, numpy 2, scikit-learn 1.9, each pinned to a tested major range.
- added a pip-audit dependency scan in ci.
- removed the stale `notebooks/` analysis; evaluation lives in `scripts/evaluate.py`.

## 2026-06-06

- `bookrec` package: a validated csv loader, an in-memory tf-idf recommender, and a
  `create_app` factory.
- a ~1,500-book catalog (a subset of best books ever) with build and evaluation scripts,
  packaging, a lockfile, and ci.
- book detail pages read ids as strings so url lookups match; search treats the query
  literally; recommendations are built in memory at startup.
