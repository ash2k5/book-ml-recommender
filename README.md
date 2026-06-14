# Book Recommender API

A content-based book recommender served as a JSON API. It vectorizes each book's
title, author, genres, and description with TF-IDF and ranks similar books by
cosine similarity. FastAPI exposes the catalog, per-book recommendations, related
books, and search, with an OpenAPI schema for typed clients.

## How it works

1. The catalog (`data/books.csv`) is loaded and validated at startup.
2. Each book's text fields are combined and vectorized with TF-IDF (unigrams and
   bigrams, English stop words).
3. Recommendations for a book are its nearest neighbors by cosine similarity,
   computed in memory. Related books are others in the same genre or by the same
   author.

## Dataset

`data/books.csv` is a ~1,500-book subset of the Best Books Ever dataset, chosen for
description quality and genre spread. The application code is MIT licensed; the
dataset is CC BY-NC 4.0, with attribution in [data/README.md](data/README.md).
Regenerate the subset with:

    uv run --extra collect python scripts/build_dataset.py

## Quick start

    uv sync
    uv run python -m bookrec          # dev server at http://127.0.0.1:8000

Run the server directly, or with Docker:

    uv run uvicorn bookrec.api:app --port 8000
    docker build -t bookrec . && docker run -p 8000:8000 bookrec

Interactive docs are at `/docs`; the OpenAPI schema is at `/openapi.json`.

## API

    GET /books?genre=&limit=             featured books, or filtered by genre
    GET /books/{id}                      a single book (404 if absent)
    GET /books/{id}/recommendations?n=   TF-IDF nearest neighbours, with scores
    GET /books/{id}/related              same-genre and same-author books
    GET /genres                          all genres, sorted
    GET /search?q=&limit=                literal title/author substring search
    GET /healthz                         health check and catalog size

`ALLOWED_ORIGINS` (comma-separated) restricts CORS to known frontends; it defaults
to `*` for a public read-only API. `DATA_PATH` overrides the catalog location.

## Tests

    uv run ruff check .
    uv run pytest

## Evaluation

`scripts/evaluate.py` reports genre/author consistency and writes a summary figure
to `models/model_analysis.png`:

    uv run --extra viz python scripts/evaluate.py

## Project structure

    bookrec/         package: data loader, recommender, FastAPI app
    data/books.csv   committed book catalog
    scripts/         offline dataset build and evaluation
    tests/           pytest suite
