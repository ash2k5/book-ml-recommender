"""Offline evaluation of the content-based recommender.

Reports how often a book's top recommendations share its genre and author, and
writes a summary figure to models/model_analysis.png. Run with the viz extra:

    uv run --extra viz python scripts/evaluate.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tome.data import load_books
from tome.recommender import BookRecommender

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "books.csv"
OUT_PATH = REPO_ROOT / "models" / "model_analysis.png"
TOP_N = 5


def consistency(recommender: BookRecommender, books) -> tuple[float, float]:
    genre_scores: list[float] = []
    author_scores: list[float] = []
    for book in books.itertuples():
        recs = recommender.recommend(book.id, n=TOP_N)
        if not recs:
            continue
        genre_scores.append(sum(r["genre"] == book.genre for r in recs) / len(recs))
        author_scores.append(sum(r["author"] == book.author for r in recs) / len(recs))
    return float(np.mean(genre_scores)), float(np.mean(author_scores))


def main() -> None:
    plt.switch_backend("Agg")
    books = load_books(DATA_PATH)
    recommender = BookRecommender(books)
    genre_c, author_c = consistency(recommender, books)

    print(f"books: {len(books)}")
    print(f"genre consistency (top {TOP_N}): {genre_c:.3f}")
    print(f"author consistency (top {TOP_N}): {author_c:.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(books["rating"], bins=20, color="#667eea", edgecolor="white")
    axes[0, 0].set_title("Rating distribution")
    axes[0, 0].set_xlabel("Goodreads rating")

    top_genres = books["genre"].value_counts().head(12)
    axes[0, 1].barh(top_genres.index[::-1], top_genres.to_numpy()[::-1], color="#764ba2")
    axes[0, 1].set_title("Top genres")

    years = books.loc[books["year"] > 0, "year"]
    axes[1, 0].hist(years, bins=30, color="#10b981", edgecolor="white")
    axes[1, 0].set_title("Publication year")

    scores = [genre_c, author_c]
    axes[1, 1].bar(["Genre", "Author"], scores, color=["#f093fb", "#f5576c"])
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title(f"Recommendation consistency (top {TOP_N})")
    for i, value in enumerate(scores):
        axes[1, 1].text(i, value + 0.02, f"{value:.2f}", ha="center")

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=120, bbox_inches="tight")
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
