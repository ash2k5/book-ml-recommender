"""Tests for the content-based recommender."""

from bookrec.data import load_books
from bookrec.recommender import BookRecommender


def test_recommend_finds_most_similar(recommender):
    recs = recommender.recommend("1")
    assert recs[0]["id"] == "2"  # same series shares the most terms
    assert all(rec["id"] != "1" for rec in recs)  # excludes the book itself


def test_recommend_unknown_id_returns_empty(recommender):
    assert recommender.recommend("999") == []


def test_recommend_caps_at_available(recommender):
    assert len(recommender.recommend("1", n=10)) == 3


def test_recommend_attaches_similarity_score(recommender):
    recs = recommender.recommend("1")
    assert all(0.0 <= rec["similarity_score"] <= 1.0 for rec in recs)


def test_single_book_corpus_returns_no_recs(write_catalog, sample_rows):
    rec = BookRecommender(load_books(write_catalog(sample_rows[:1], "one.csv")))
    assert rec.recommend("1") == []


def test_search_treats_query_literally(recommender):
    assert recommender.search("(") == []  # regex metachar must not crash


def test_search_matches_title_and_author(recommender):
    assert {b["id"] for b in recommender.search("space")} == {"1", "2"}
    assert {b["id"] for b in recommender.search("nova")} == {"1", "2"}


def test_search_is_case_insensitive(recommender):
    assert {b["id"] for b in recommender.search("SPACE")} == {"1", "2"}


def test_search_blank_query_returns_empty(recommender):
    assert recommender.search("") == []
    assert recommender.search("   ") == []


def test_by_genre_excludes_self(recommender):
    assert {b["id"] for b in recommender.by_genre("Science Fiction", "1")} == {"2"}


def test_by_author_excludes_self(recommender):
    assert {b["id"] for b in recommender.by_author("A. Nova", "1")} == {"2"}


def test_get_returns_json_safe_native_types(recommender):
    book = recommender.get("1")
    assert isinstance(book["year"], int)
    assert isinstance(book["rating"], float)
    assert isinstance(book["num_ratings"], int)
    assert "combined_features" not in book


def test_get_unknown_returns_none(recommender):
    assert recommender.get("999") is None


def test_genres_sorted_and_unique(recommender):
    assert recommender.genres() == ["Mystery", "Romance", "Science Fiction"]


def test_featured_respects_limit(recommender):
    assert len(recommender.featured(2)) == 2


def test_in_genre_returns_matches(recommender):
    assert {b["id"] for b in recommender.in_genre("Science Fiction")} == {"1", "2"}


def test_len_reports_catalog_size(recommender):
    assert len(recommender) == 4
