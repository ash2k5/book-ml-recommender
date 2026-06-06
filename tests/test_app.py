"""Tests for the Flask routes."""

import pytest

from bookrec.app import create_app


def test_index_lists_books(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Space Saga One" in response.data


def test_index_genre_filter(client):
    response = client.get("/?genre=Romance")
    assert response.status_code == 200
    assert b"Quiet Romance" in response.data
    assert b"Space Saga One" not in response.data


def test_book_detail_renders_with_recommendations(client):
    response = client.get("/book/1")
    assert response.status_code == 200
    assert b"Space Saga One" in response.data
    assert b"Space Saga Two" in response.data


def test_book_detail_unknown_returns_404(client):
    response = client.get("/book/999")
    assert response.status_code == 404
    assert b"Book not found" in response.data


def test_search_returns_json(client):
    response = client.get("/search?q=space")
    assert response.status_code == 200
    assert {book["id"] for book in response.get_json()} == {"1", "2"}


def test_search_regex_metacharacter_does_not_500(client):
    response = client.get("/search?q=(")
    assert response.status_code == 200
    assert response.get_json() == []


def test_api_recommendations_returns_json(client):
    response = client.get("/api/recommendations/1")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload[0]["id"] == "2"


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "books": 4}


def test_missing_data_file_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_app(tmp_path / "absent.csv")
