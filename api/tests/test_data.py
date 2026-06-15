"""Tests for the catalog loader and validation."""

import pytest

from bookrec.data import REQUIRED_COLUMNS, load_books


def test_loads_all_rows(books):
    assert len(books) == 4
    assert "combined_features" in books.columns


def test_combined_features_includes_text(books):
    combined = books[books["id"] == "1"].iloc[0]["combined_features"]
    assert "Space Saga One" in combined
    assert "A. Nova" in combined
    assert "starships" in combined


def test_id_coerced_to_string(write_catalog, sample_rows):
    for row in sample_rows:
        row["id"] = int(row["id"])
    df = load_books(write_catalog(sample_rows, "int_ids.csv"))
    assert df["id"].map(type).eq(str).all()
    assert "1" in set(df["id"])


def test_missing_column_raises(write_catalog, sample_rows):
    rows = [{k: v for k, v in row.items() if k != "rating"} for row in sample_rows]
    with pytest.raises(ValueError, match="missing required columns"):
        load_books(write_catalog(rows, "missing.csv"))


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_books(tmp_path / "nope.csv")


def test_dedupes_on_id(write_catalog, sample_rows):
    rows = [*sample_rows, dict(sample_rows[0], title="Duplicate")]
    df = load_books(write_catalog(rows, "dup.csv"))
    assert (df["id"] == "1").sum() == 1
    assert df[df["id"] == "1"].iloc[0]["title"] == "Space Saga One"


def test_drops_empty_title(write_catalog, sample_rows):
    rows = [*sample_rows, dict(sample_rows[0], id="9", title="   ")]
    df = load_books(write_catalog(rows, "empty_title.csv"))
    assert "9" not in set(df["id"])


def test_numeric_columns_coerced(books):
    assert books["year"].dtype.kind == "i"
    assert books["num_ratings"].dtype.kind == "i"
    assert int(books[books["id"] == "4"].iloc[0]["year"]) == 0


def test_unparseable_numbers_become_zero(write_catalog, sample_rows):
    sample_rows[0]["year"] = "n/a"
    sample_rows[0]["num_ratings"] = ""
    df = load_books(write_catalog(sample_rows, "bad_numbers.csv"))
    row = df[df["id"] == "1"].iloc[0]
    assert int(row["year"]) == 0
    assert int(row["num_ratings"]) == 0


def test_empty_catalog_raises(write_catalog, sample_rows):
    rows = [dict(sample_rows[0], id="", title="")]
    with pytest.raises(ValueError, match="no usable rows"):
        load_books(write_catalog(rows, "empty.csv"))


def test_required_columns_present():
    assert {"id", "title", "description", "cover_url"} <= set(REQUIRED_COLUMNS)
