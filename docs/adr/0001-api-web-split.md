# 1. split api/ and web/, deploy independently

- Status: accepted
- Date: 2026-06-20

## Context
The recommender is Python (FastAPI + scikit-learn TF-IDF); the UI is Next.js.

## Decision
One repo, two deployables: `api/` to Render (`render.yaml`, `rootDir: api`) and `web/` to Vercel
(Root Directory `web`). CI is path-filtered (`api/**`, `web/**`). The web client is typed from the
API's OpenAPI schema (`npm run gen:api`).

## Consequences
Independent deploys. An API contract change means regenerating the web types.
