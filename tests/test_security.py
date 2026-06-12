"""Regression tests for the XSS hardening of cards and search results.

The template side is asserted through the rendered markup. The client-side
``displaySearchResults`` DOM construction is guarded statically here and verified
manually against a payload fixture, since this Python repo ships no JS test runner.
"""

from pathlib import Path

from bookrec.app import create_app

PAYLOAD_ROW = {
    "id": "p1\"'<x>",
    "title": "<script>alert('xss')</script>",
    "author": 'Mallory "the<img>" Doe',
    "genre": "Mystery",
    "genres": "Mystery",
    "description": "payload book",
    "rating": 4.0,
    "num_ratings": 10,
    "year": 2020,
    "publisher": "P",
    "cover_url": "",
}


def test_book_cards_use_data_attribute(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b'data-book-id="1"' in response.data
    assert b'onclick="viewBook' not in response.data


def test_recommendation_cards_use_data_attribute(client):
    response = client.get("/book/1")
    assert response.status_code == 200
    assert b"data-book-id=" in response.data
    assert b"viewBook(" not in response.data


def test_catalog_markup_in_fields_is_escaped(write_catalog):
    client = create_app(write_catalog([PAYLOAD_ROW])).test_client()
    response = client.get("/")
    assert response.status_code == 200
    # Title markup is HTML-escaped, not rendered as a live tag.
    assert b"<script>alert(" not in response.data
    assert b"&lt;script&gt;" in response.data
    # The quote/markup-bearing id stays inside an escaped data attribute.
    assert b"data-book-id=" in response.data
    assert b"<x>" not in response.data
    assert b'onclick="viewBook' not in response.data


def test_search_script_avoids_innerhtml_sink():
    src = (Path(__file__).resolve().parents[1] / "static" / "script.js").read_text(encoding="utf-8")
    assert "createElement" in src
    assert "viewBook" not in src
    assert "innerHTML = html" not in src
