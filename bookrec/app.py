"""Flask application factory for the book recommender."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from bookrec.data import load_books
from bookrec.recommender import BookRecommender

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA = _REPO_ROOT / "data" / "books.csv"


def create_app(data_path: str | os.PathLike | None = None) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(_REPO_ROOT / "templates"),
        static_folder=str(_REPO_ROOT / "static"),
    )
    app.config["DEBUG"] = os.environ.get("FLASK_DEBUG", "").lower() in {
        "1",
        "true",
        "yes",
    }

    path = Path(data_path or os.environ.get("DATA_PATH") or _DEFAULT_DATA)
    recommender = BookRecommender(load_books(path))
    app.config["recommender"] = recommender

    @app.context_processor
    def inject_globals():
        return {"current_year": datetime.now(UTC).year}

    @app.route("/")
    def index():
        genre = request.args.get("genre", "").strip()
        books = recommender.in_genre(genre) if genre else recommender.featured(60)
        return render_template(
            "index.html",
            books=books,
            genres=recommender.genres(),
            selected_genre=genre,
            total=len(recommender),
        )

    @app.route("/book/<book_id>")
    def book_detail(book_id: str):
        book = recommender.get(book_id)
        if book is None:
            return render_template("404.html", book_id=book_id), 404
        return render_template(
            "book_detail.html",
            book=book,
            recommendations=recommender.recommend(book_id, n=6),
            same_genre_books=recommender.by_genre(book["genre"], book_id, n=4),
            same_author_books=recommender.by_author(book["author"], book_id, n=3),
            current_genre=book["genre"],
            current_author=book["author"],
        )

    @app.route("/api/recommendations/<book_id>")
    def api_recommendations(book_id: str):
        return jsonify(recommender.recommend(book_id))

    @app.route("/search")
    def search():
        return jsonify(recommender.search(request.args.get("q", "")))

    @app.route("/healthz")
    def healthz():
        return jsonify(status="ok", books=len(recommender))

    return app
