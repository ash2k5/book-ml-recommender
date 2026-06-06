# Book Recommender

A content-based book recommender. It vectorizes each book's title, author, genres,
and description with TF-IDF and ranks similar books by cosine similarity. A Flask
app serves a browsable catalog, per-book recommendations, and a JSON search API.

## How it works

1. The catalog (`data/books.csv`) is loaded and validated at startup.
2. Each book's text fields are combined and vectorized with TF-IDF (unigrams and
   bigrams, English stop words).
3. Recommendations for a book are its nearest neighbors by cosine similarity,
   computed in memory. The detail page also shows other books in the same genre
   and by the same author.

The model is built in memory at startup (~1,500 books, well under a second), so
there are no pickled artifacts to keep in sync with the data.

## Dataset

`data/books.csv` is a ~1,500-book subset of the Best Books Ever dataset, chosen for
description quality and genre spread. The application code is MIT licensed; the
dataset is CC BY-NC 4.0, with attribution in [data/README.md](data/README.md).
Regenerate the subset with:

    uv run --extra collect python scripts/build_dataset.py

## Quick start

    uv sync
    uv run python -m bookrec          # dev server at http://127.0.0.1:5000

Run with a production server, or with Docker:

    uv run gunicorn wsgi:app          # Linux/macOS
    docker build -t bookrec . && docker run -p 8000:8000 bookrec

## Tests

    uv run ruff check .
    uv run pytest

## Evaluation

`scripts/evaluate.py` reports genre/author consistency and writes a summary figure
to `models/model_analysis.png`:

    uv run --extra viz python scripts/evaluate.py

## Deployment

`render.yaml` defines a free-tier Render web service built from the `Dockerfile`.
The app reads `$PORT` and exposes `/healthz` for health checks. The committed
catalog ships in the image, so no runtime data source or secret is required.

## Project structure

    bookrec/         package: data loader, recommender, Flask app factory
    templates/       Jinja templates
    static/          CSS and JS
    data/books.csv   committed book catalog
    scripts/         offline dataset build and evaluation
    tests/           pytest suite
    wsgi.py          gunicorn entry point

## Routes

    GET /                          featured books, genre filter
    GET /book/<id>                 book detail and recommendations
    GET /search?q=<query>          JSON title/author search
    GET /api/recommendations/<id>  JSON recommendations
    GET /healthz                   health check
