# book ml recommender

a book recommender over a ~1,500 title catalog. it ranks books by content, not ratings:
each book's plot, author, and genres are combined and vectorized with tf-idf, then ranked
by cosine similarity to the one you're viewing, so the matches reflect what a book is
about. browse a featured shelf, filter by genre, search, and open any title for
recommendations plus more from the same genre and author.

https://book-ml-web.vercel.app

## run locally

a python api (`api/`) and a next.js frontend (`web/`).

api, on http://localhost:8000:

```bash
cd api
uv sync --extra dev
uv run uvicorn bookrec.api:app --reload
```

web, on http://localhost:3000 (set `API_BASE_URL` to your local api):

```bash
cd web
npm ci
npm run dev
```

## tests

```bash
cd api && uv run pytest
cd web && npm test
```

## data

the catalog is a ~1,500-book subset of the best books ever dataset (cc by-nc 4.0); see
`api/data/README.md` for attribution. code is MIT (`LICENSE`).
