"""Book catalog loading, validation, and normalization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = (
    "id",
    "title",
    "author",
    "genre",
    "genres",
    "description",
    "rating",
    "num_ratings",
    "year",
    "publisher",
    "cover_url",
)

_TEXT_COLUMNS = (
    "title",
    "author",
    "genre",
    "genres",
    "description",
    "publisher",
    "cover_url",
)


def load_books(path: str | Path) -> pd.DataFrame:
    """Load the book catalog from CSV and return a validated DataFrame.

    The ``id`` column is read as a string so URL lookups match regardless of
    whether the source ids are numeric. A ``combined_features`` column is added
    for the recommender to vectorize.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"book catalog not found: {path}")

    df = pd.read_csv(path, dtype={"id": str})

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"catalog missing required columns: {', '.join(missing)}")

    df = df[list(REQUIRED_COLUMNS)].copy()

    for column in _TEXT_COLUMNS:
        df[column] = df[column].fillna("").astype(str).str.strip()

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["num_ratings"] = pd.to_numeric(df["num_ratings"], errors="coerce").fillna(0).astype(int)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)

    df = df[df["id"].str.len() > 0]
    df = df[df["title"].str.len() > 0]
    df = df.drop_duplicates(subset="id", keep="first").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"catalog has no usable rows: {path}")

    df["combined_features"] = (
        df["title"] + " " + df["author"] + " " + df["genres"] + " " + df["description"]
    ).str.strip()

    return df
