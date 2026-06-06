"""Content-based book recommender using TF-IDF and cosine similarity."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_NUMERIC_INT_FIELDS = ("num_ratings", "year")


class BookRecommender:
    """Builds an in-memory TF-IDF model over a book catalog.

    The matrix is built once at construction so its rows always align with the
    catalog, removing the stale-pickle mismatch the original code was prone to.
    """

    def __init__(self, books: pd.DataFrame) -> None:
        if "combined_features" not in books.columns:
            raise ValueError("books frame must include a 'combined_features' column")

        self.books = books.reset_index(drop=True)
        self._position_by_id = {book_id: i for i, book_id in enumerate(self.books["id"])}
        self._vectorizer = TfidfVectorizer(
            max_features=20000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self._matrix = self._vectorizer.fit_transform(self.books["combined_features"])

    def __len__(self) -> int:
        return len(self.books)

    def all_books(self) -> list[dict]:
        return self._records(self.books)

    def featured(self, limit: int = 60) -> list[dict]:
        """Return the most popular books (catalog is ordered by popularity)."""
        return self._records(self.books.head(limit))

    def in_genre(self, genre: str) -> list[dict]:
        return self._records(self.books[self.books["genre"] == genre])

    def genres(self) -> list[str]:
        return sorted({genre for genre in self.books["genre"] if genre})

    def get(self, book_id: str) -> dict | None:
        position = self._position_by_id.get(book_id)
        return None if position is None else self._record(position)

    def recommend(self, book_id: str, n: int = 6) -> list[dict]:
        """Return the ``n`` most similar books, excluding the book itself."""
        position = self._position_by_id.get(book_id)
        if position is None:
            return []

        scores = cosine_similarity(self._matrix[position], self._matrix).ravel()
        results: list[dict] = []
        for index in scores.argsort()[::-1]:
            if index == position:
                continue
            record = self._record(int(index))
            record["similarity_score"] = round(float(scores[index]), 3)
            results.append(record)
            if len(results) >= n:
                break
        return results

    def by_genre(self, genre: str, exclude_id: str, n: int = 4) -> list[dict]:
        mask = (self.books["genre"] == genre) & (self.books["id"] != exclude_id)
        return self._records(self.books[mask].head(n))

    def by_author(self, author: str, exclude_id: str, n: int = 3) -> list[dict]:
        mask = (self.books["author"] == author) & (self.books["id"] != exclude_id)
        return self._records(self.books[mask].head(n))

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """Substring search over title and author. Treats the query literally."""
        query = (query or "").strip()
        if not query:
            return []
        title = self.books["title"].str.contains(query, case=False, na=False, regex=False)
        author = self.books["author"].str.contains(query, case=False, na=False, regex=False)
        return self._records(self.books[title | author].head(limit))

    def _record(self, position: int) -> dict:
        return self._coerce(self.books.iloc[position].to_dict())

    def _records(self, frame: pd.DataFrame) -> list[dict]:
        return [self._coerce(record) for record in frame.to_dict("records")]

    @staticmethod
    def _coerce(record: dict) -> dict:
        """Convert numpy scalars to native types so records are JSON-safe."""
        record.pop("combined_features", None)
        rating = record.get("rating")
        record["rating"] = round(float(rating), 2) if pd.notna(rating) else None
        for field in _NUMERIC_INT_FIELDS:
            value = record.get(field)
            record[field] = int(value) if pd.notna(value) else None
        return record
