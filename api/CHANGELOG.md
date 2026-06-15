## 2026-06-14

### Changed
- Replaced the Flask + Jinja web layer with a FastAPI JSON API (`bookrec/api.py`).
  Endpoints: `/books`, `/books/{id}`, `/books/{id}/recommendations`,
  `/books/{id}/related`, `/genres`, `/search`, `/healthz`, plus an OpenAPI schema
  at `/openapi.json`. The TF-IDF recommender and dataset are unchanged, so
  recommendations are identical to the previous release.
- Query parameters are bounded (`limit`, `n`) and validated at the API boundary;
  unknown book ids now return 404 rather than an empty list.
- Production server is `uvicorn`; `python -m bookrec` runs the dev server on 8000.

### Added
- CORS support, allow-listed via the `ALLOWED_ORIGINS` environment variable
  (defaults to `*` for this public read-only API).

### Removed
- Server-rendered templates and static assets (the frontend moves to a separate
  Next.js app), along with their DOM-XSS regression tests and the `gunicorn`/WSGI
  entry point. Dropped Flask in favour of FastAPI + Uvicorn.

## 2026-06-12

### Security
- Search results are rebuilt with DOM APIs and book cards navigate through a
  delegated `data-book-id` listener, replacing the field-into-`innerHTML` and
  inline `onclick` interpolation that was a latent DOM-XSS sink.

### Changed
- Moved the scientific stack to pandas 3, numpy 2, and scikit-learn 1.9, and
  pinned each dependency to a tested major range (`pandas>=3,<4`, `numpy>=2.1,<3`,
  `scikit-learn>=1.9,<2`, `flask>=3.1,<4`) so `uv lock --upgrade` cannot cross a
  major on its own.

### Added
- `pip-audit` dependency scan in CI.

### Removed
- Stale `notebooks/` analysis from the original pickle workflow; evaluation now
  lives in `scripts/evaluate.py`.

## 2026-06-06

### Added
- `bookrec` package: a validated CSV loader, an in-memory TF-IDF recommender, and
  a `create_app` factory with a WSGI entry point.
- A ~1,500-book catalog (a subset of Best Books Ever) with a build script, an
  evaluation script, packaging, a lockfile, Docker/Render config, and CI.

### Fixed
- Book detail pages no longer 404: ids are read as strings so URL lookups match.
- Data loads under WSGI, not only when run as `__main__`.
- Search treats the query literally, so regex metacharacters can't crash it.

### Changed
- Recommendations are built in memory at startup, removing the pickled artifacts
  that could drift from the data.
