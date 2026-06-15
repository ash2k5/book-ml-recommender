"""Shared test fixtures."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from bookrec.api import create_app
from bookrec.data import load_books
from bookrec.recommender import BookRecommender

SAMPLE_ROWS = [
    {
        "id": "1",
        "title": "Space Saga One",
        "author": "A. Nova",
        "genre": "Science Fiction",
        "genres": "Science Fiction, Space Opera",
        "description": "epic space opera about starships galactic empires and aliens",
        "rating": 4.5,
        "num_ratings": 1000,
        "year": 2001,
        "publisher": "Orbit",
        "cover_url": "http://example.com/1.jpg",
    },
    {
        "id": "2",
        "title": "Space Saga Two",
        "author": "A. Nova",
        "genre": "Science Fiction",
        "genres": "Science Fiction, Space Opera",
        "description": "more starships galactic empires aliens and space battles",
        "rating": 4.2,
        "num_ratings": 800,
        "year": 2003,
        "publisher": "Orbit",
        "cover_url": "",
    },
    {
        "id": "3",
        "title": "Quiet Romance",
        "author": "B. Heart",
        "genre": "Romance",
        "genres": "Romance, Contemporary",
        "description": "a tender love story about marriage and relationships in a small town",
        "rating": 3.9,
        "num_ratings": 500,
        "year": 2010,
        "publisher": "Harlequin",
        "cover_url": "",
    },
    {
        "id": "4",
        "title": "Mystery Manor",
        "author": "C. Clue",
        "genre": "Mystery",
        "genres": "Mystery, Thriller",
        "description": "a detective investigates a murder in an old manor full of suspects",
        "rating": 4.0,
        "num_ratings": 0,
        "year": 0,
        "publisher": "",
        "cover_url": "",
    },
]


@pytest.fixture
def sample_rows():
    return [dict(row) for row in SAMPLE_ROWS]


@pytest.fixture
def write_catalog(tmp_path):
    def _write(rows, name="books.csv"):
        path = tmp_path / name
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    return _write


@pytest.fixture
def catalog_path(write_catalog):
    return write_catalog(SAMPLE_ROWS)


@pytest.fixture
def books(catalog_path):
    return load_books(catalog_path)


@pytest.fixture
def recommender(books):
    return BookRecommender(books)


@pytest.fixture
def client(catalog_path):
    with TestClient(create_app(catalog_path)) as test_client:
        yield test_client
