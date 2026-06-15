# Book ML Recommender

A full-stack book recommendation app over a 1,500-title catalog. The backend ranks
books by content similarity (TF-IDF over plot, author, and genre); the frontend is an
editorial reading experience built on a custom design system.

- **Web app:** _Vercel deploy (link added on deploy)_
- **API:** https://book-ml-recommender.onrender.com — [OpenAPI schema](https://book-ml-recommender.onrender.com/openapi.json)

## What it does

Browse a featured shelf, filter by genre, open any title for its details, and get
"you might also like" recommendations plus more from the same genre and author. Search
runs over titles and authors.

Recommendations are **content-based**, not rating averages: each book's plot summary,
author, and genres are combined into one text field, vectorized with TF-IDF, and ranked
by cosine similarity to the book you're viewing. Matches reflect what a book is *about*.
For example, *The Hunger Games* surfaces *Catching Fire* and *Mockingjay*.

## Architecture

```
book-ml-recommender/
├── api/   FastAPI JSON API + scikit-learn TF-IDF recommender   ->  Render (Docker)
└── web/   Next.js App Router frontend on @ash2k5/cinematic-ds  ->  Vercel
```

The two halves deploy independently. The frontend's API client is **typed from the
backend's OpenAPI schema** (`web/app/lib/schema.d.ts`, regenerated with `npm run gen:api`),
so the contract between them is checked at compile time.

| | Stack |
|---|---|
| api | Python 3.12, FastAPI, scikit-learn, pandas, uv; pytest + ruff; Docker on Render |
| web | Next.js 16 (App Router, server components), React 19, TypeScript, Tailwind v4, the [`@ash2k5/cinematic-ds`](https://github.com/ash2k5/design-system) design system; Vitest; Vercel |

## Run locally

**API** (http://localhost:8000, `/docs` for Swagger):

```bash
cd api
uv sync --extra dev
uv run uvicorn bookrec.api:app --reload
```

**Web** (http://localhost:3000) — install needs a classic `read:packages` token for the
design system package (GitHub Packages requires auth even for public packages):

```bash
cd web
export NODE_AUTH_TOKEN=<classic PAT with read:packages>
npm ci
npm run dev   # set API_BASE_URL to point elsewhere; defaults to the live API
```

## Tests

```bash
cd api && uv run pytest        # data, recommender, and API endpoint tests
cd web && npm test             # API client + component tests
```

## Data

Catalog is a ~1,500-book subset of the **Best Books Ever** dataset (CC BY-NC 4.0); see
`api/data/README.md` for attribution. Application code is MIT (`LICENSE`).
