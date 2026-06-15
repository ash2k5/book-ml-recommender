"""Tests for the FastAPI JSON endpoints."""

import pytest
from fastapi.testclient import TestClient

from bookrec.api import create_app


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "books": 4}


def test_genres_sorted(client):
    response = client.get("/genres")
    assert response.status_code == 200
    assert response.json() == ["Mystery", "Romance", "Science Fiction"]


def test_books_returns_featured(client):
    response = client.get("/books")
    assert response.status_code == 200
    ids = {book["id"] for book in response.json()}
    assert ids == {"1", "2", "3", "4"}


def test_books_genre_filter(client):
    response = client.get("/books", params={"genre": "Romance"})
    assert response.status_code == 200
    ids = {book["id"] for book in response.json()}
    assert ids == {"3"}


def test_books_limit(client):
    response = client.get("/books", params={"limit": 2})
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_books_limit_out_of_range_is_422(client):
    assert client.get("/books", params={"limit": 0}).status_code == 422
    assert client.get("/books", params={"limit": 999}).status_code == 422


def test_book_detail(client):
    response = client.get("/books/1")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "1"
    assert body["title"] == "Space Saga One"
    assert "combined_features" not in body


def test_book_unknown_returns_404(client):
    response = client.get("/books/999")
    assert response.status_code == 404
    assert response.json()["detail"] == "book not found"


def test_recommendations(client):
    response = client.get("/books/1/recommendations")
    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["id"] == "2"
    assert all(book["id"] != "1" for book in payload)
    assert all(0.0 <= book["similarity_score"] <= 1.0 for book in payload)


def test_recommendations_caps_at_available(client):
    response = client.get("/books/1/recommendations", params={"n": 10})
    assert response.status_code == 200
    assert len(response.json()) == 3


def test_recommendations_unknown_returns_404(client):
    assert client.get("/books/999/recommendations").status_code == 404


def test_recommendations_n_out_of_range_is_422(client):
    assert client.get("/books/1/recommendations", params={"n": 0}).status_code == 422


def test_related_same_genre_and_author(client):
    response = client.get("/books/1/related")
    assert response.status_code == 200
    body = response.json()
    assert {book["id"] for book in body["same_genre"]} == {"2"}
    assert {book["id"] for book in body["same_author"]} == {"2"}


def test_related_unknown_returns_404(client):
    assert client.get("/books/999/related").status_code == 404


def test_search_matches_title_and_author(client):
    response = client.get("/search", params={"q": "space"})
    assert response.status_code == 200
    assert {book["id"] for book in response.json()} == {"1", "2"}


def test_search_regex_metacharacter_does_not_500(client):
    response = client.get("/search", params={"q": "("})
    assert response.status_code == 200
    assert response.json() == []


def test_search_blank_returns_empty(client):
    assert client.get("/search", params={"q": ""}).json() == []


def test_search_limit(client):
    response = client.get("/search", params={"q": "space", "limit": 1})
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_openapi_schema_is_served(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["version"] == "2.0.0"
    assert "/books/{book_id}" in schema["paths"]


def test_cors_header_present(client):
    response = client.get("/healthz", headers={"Origin": "http://localhost:3000"})
    assert response.headers["access-control-allow-origin"] == "*"


def test_cors_allowlist_from_env(monkeypatch, catalog_path):
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://app.example.com")
    with TestClient(create_app(catalog_path)) as scoped:
        response = scoped.get("/healthz", headers={"Origin": "https://app.example.com"})
    assert response.headers["access-control-allow-origin"] == "https://app.example.com"


def test_missing_data_file_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_app(tmp_path / "absent.csv")
