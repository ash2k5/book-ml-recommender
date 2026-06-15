"""Build the committed book catalog from the Best Books Ever dataset.

Downloads the Best Books Ever dataset (Casanova Lozano & Costa Planells, 2020,
CC BY-NC 4.0, Zenodo 10.5281/zenodo.4265096), cleans it, and writes a diverse
subset to ``data/books.csv``. The raw download is cached under ``data/raw/``
(gitignored); only the derived subset is committed.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd
import requests

SOURCE_URL = (
    "https://raw.githubusercontent.com/scostap/goodreads_bbe_dataset/"
    "main/Best_Books_Ever_dataset/books_1.Best_Books_Ever.csv"
)
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = REPO_ROOT / "data" / "raw" / "bbe.csv"
OUT_PATH = REPO_ROOT / "data" / "books.csv"

TARGET_ROWS = 1500
PER_GENRE_CAP = 110
MIN_DESCRIPTION = 60
_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20[0-2]\d)\b")


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"using cached {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)


def _text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip() if pd.notna(value) else ""


def clean_author(value: object) -> str:
    first = str(value).split(",")[0]
    return re.sub(r"\s*\([^)]*\)", "", first).strip()


def parse_genres(value: object) -> list[str]:
    text = str(value).strip()
    if not text:
        return []
    try:
        items = ast.literal_eval(text)
        if isinstance(items, (list, tuple)):
            return [str(genre).strip() for genre in items if str(genre).strip()]
    except (ValueError, SyntaxError):
        pass
    return [genre.strip() for genre in text.split(",") if genre.strip()]


def _year_from_date(value: object) -> int:
    text = str(value)
    match = re.search(r"\b\d{1,2}/\d{1,2}/(\d{2})\b", text)
    if match:
        yy = int(match.group(1))
        return 2000 + yy if yy <= 21 else 1900 + yy
    match = _YEAR_RE.search(text)
    return int(match.group(0)) if match else 0


def resolve_year(first_published: object, published: object) -> int:
    """Resolve a publication year from BBE's two-digit dates.

    Prefers the first-publication date. Two-digit years are ambiguous for old
    classics, so a first-publication year that postdates the edition (logically
    impossible) or lands in the future is treated as unknown rather than shown
    wrong.
    """
    first = _year_from_date(first_published)
    edition = _year_from_date(published)
    if first and edition and first > edition:
        return 0
    year = first or edition
    return 0 if year > 2021 else year


def _to_int(value: object) -> int:
    number = pd.to_numeric(value, errors="coerce")
    return int(number) if pd.notna(number) else 0


def build() -> pd.DataFrame:
    raw = pd.read_csv(RAW_PATH)
    raw["bbeScore"] = pd.to_numeric(raw["bbeScore"], errors="coerce").fillna(0)
    raw = raw.sort_values("bbeScore", ascending=False)

    seen: set[tuple[str, str]] = set()
    per_genre: dict[str, int] = {}
    rows: list[dict] = []

    for _, book in raw.iterrows():
        title = _text(book.get("title"))
        author = clean_author(book.get("author"))
        description = _text(book.get("description"))
        rating = pd.to_numeric(book.get("rating"), errors="coerce")
        genres = parse_genres(book.get("genres"))

        if not title or not author or len(description) < MIN_DESCRIPTION:
            continue
        if pd.isna(rating) or rating <= 0 or not genres:
            continue

        key = (title.lower(), author.lower())
        if key in seen:
            continue
        primary = genres[0]
        if per_genre.get(primary, 0) >= PER_GENRE_CAP:
            continue

        cover = _text(book.get("coverImg"))
        rows.append(
            {
                "id": str(len(rows) + 1),
                "title": title,
                "author": author,
                "genre": primary,
                "genres": ", ".join(genres[:5]),
                "description": description,
                "rating": round(float(rating), 2),
                "num_ratings": _to_int(book.get("numRatings")),
                "year": resolve_year(book.get("firstPublishDate"), book.get("publishDate")),
                "publisher": _text(book.get("publisher")),
                "cover_url": "" if "nophoto" in cover else cover,
            }
        )
        seen.add(key)
        per_genre[primary] = per_genre.get(primary, 0) + 1
        if len(rows) >= TARGET_ROWS:
            break

    return pd.DataFrame(rows)


def main() -> None:
    download(SOURCE_URL, RAW_PATH)
    books = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    books.to_csv(OUT_PATH, index=False)
    print(f"wrote {len(books)} books to {OUT_PATH.relative_to(REPO_ROOT)}")
    print(
        f"genres: {books['genre'].nunique()} | years: {books['year'].min()}-{books['year'].max()}"
    )


if __name__ == "__main__":
    main()
