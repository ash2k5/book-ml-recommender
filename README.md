# Book ML Recommender

A book recommender over a ~1,500-title catalog. It ranks books by content, not ratings:
each book's plot, author, and genres are combined and vectorized with TF-IDF, then ranked by
cosine similarity to the one you're viewing, so the matches reflect what a book is about.
Browse a featured shelf, filter by genre, search, and open any title for recommendations
plus more from the same genre and author.

https://book-ml-web.vercel.app

## Run locally

A Python API (`api/`) and a Next.js frontend (`web/`).

API, on http://localhost:8000:

```bash
cd api
uv sync --extra dev
uv run uvicorn bookrec.api:app --reload
```

Web, on http://localhost:3000 (set `API_BASE_URL` to your local API):

```bash
cd web
npm ci
npm run dev
```

## Run with Docker

Runs the whole app in containers, no local Python or Node setup needed (Docker Desktop must be running):

```bash
docker compose up --build
```

Production build:

```bash
docker compose -f compose.prod.yaml up --build
```

## Tests

```bash
cd api && uv run pytest
cd web && npm test
```

## Data

The catalog is a ~1,500-book subset of the Best Books Ever dataset (CC BY-NC 4.0); see
`api/data/README.md` for attribution. Code is MIT (`LICENSE`).
