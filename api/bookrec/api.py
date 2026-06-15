"""FastAPI JSON API for the book recommender."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from bookrec import __version__
from bookrec.data import load_books
from bookrec.recommender import BookRecommender

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA = _REPO_ROOT / "data" / "books.csv"


class Book(BaseModel):
    id: str
    title: str
    author: str
    genre: str
    genres: str
    description: str
    rating: float | None
    num_ratings: int | None
    year: int | None
    publisher: str
    cover_url: str


class ScoredBook(Book):
    similarity_score: float


class Related(BaseModel):
    same_genre: list[Book]
    same_author: list[Book]


class Health(BaseModel):
    status: str
    books: int


def _allowed_origins() -> list[str]:
    raw = os.environ.get("ALLOWED_ORIGINS", "").strip()
    if not raw:
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def create_app(data_path: str | os.PathLike | None = None) -> FastAPI:
    path = Path(data_path or os.environ.get("DATA_PATH") or _DEFAULT_DATA)
    recommender = BookRecommender(load_books(path))

    app = FastAPI(
        title="Book ML Recommender API",
        description="Content-based book recommendations via TF-IDF and cosine similarity.",
        version=__version__,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    app.state.recommender = recommender

    def _require(book_id: str) -> dict:
        book = recommender.get(book_id)
        if book is None:
            raise HTTPException(status_code=404, detail="book not found")
        return book

    @app.get("/healthz", response_model=Health)
    def healthz() -> Health:
        return Health(status="ok", books=len(recommender))

    @app.get("/genres", response_model=list[str])
    def genres() -> list[str]:
        return recommender.genres()

    @app.get("/books", response_model=list[Book])
    def books(
        genre: str = Query("", description="Filter by exact genre; omit for featured."),
        limit: int = Query(60, ge=1, le=200),
    ) -> list[dict]:
        genre = genre.strip()
        return recommender.in_genre(genre) if genre else recommender.featured(limit)

    @app.get("/books/{book_id}", response_model=Book)
    def book(book_id: str) -> dict:
        return _require(book_id)

    @app.get("/books/{book_id}/recommendations", response_model=list[ScoredBook])
    def recommendations(book_id: str, n: int = Query(6, ge=1, le=50)) -> list[dict]:
        _require(book_id)
        return recommender.recommend(book_id, n=n)

    @app.get("/books/{book_id}/related", response_model=Related)
    def related(
        book_id: str,
        genre_limit: int = Query(4, ge=0, le=50),
        author_limit: int = Query(3, ge=0, le=50),
    ) -> Related:
        book = _require(book_id)
        return Related(
            same_genre=recommender.by_genre(book["genre"], book_id, n=genre_limit),
            same_author=recommender.by_author(book["author"], book_id, n=author_limit),
        )

    @app.get("/search", response_model=list[Book])
    def search(
        q: str = Query("", description="Substring query over title and author."),
        limit: int = Query(20, ge=1, le=100),
    ) -> list[dict]:
        return recommender.search(q, limit=limit)

    return app


app = create_app()
