"""Unit tests for storymesh.schemas.book_fetcher."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from storymesh.schemas.book_fetcher import (
    BookFetcherAgentInput,
    BookFetcherAgentOutput,
    BookRecord,
)
from storymesh.versioning.schemas import BOOK_FETCHER_SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(**overrides: object) -> BookRecord:
    defaults: dict[str, object] = dict(
        work_key="/works/OL27448W",
        title="The Lord of the Rings",
        source_genres=["fantasy"],
    )
    return BookRecord(**(defaults | overrides))


def _output(**overrides: object) -> BookFetcherAgentOutput:
    defaults: dict[str, object] = dict(
        books=[],
        queries_executed=["mystery"],
    )
    return BookFetcherAgentOutput(**(defaults | overrides))


# ---------------------------------------------------------------------------
# BookRecord
# ---------------------------------------------------------------------------

class TestBookRecord:
    def test_valid_all_fields(self) -> None:
        record = BookRecord(
            work_key="/works/OL27448W",
            title="The Lord of the Rings",
            authors=["J. R. R. Tolkien"],
            first_publish_year=1954,
            edition_count=120,
            ratings_average=4.2,
            ratings_count=850,
            subjects=["Fantasy fiction", "Middle Earth"],
            cover_id=258027,
            source_genres=["fantasy"],
        )
        assert record.work_key == "/works/OL27448W"
        assert record.title == "The Lord of the Rings"
        assert record.authors == ["J. R. R. Tolkien"]
        assert record.first_publish_year == 1954
        assert record.edition_count == 120
        assert record.ratings_average == 4.2
        assert record.ratings_count == 850
        assert record.subjects == ["Fantasy fiction", "Middle Earth"]
        assert record.cover_id == 258027
        assert record.source_genres == ["fantasy"]

    def test_valid_optional_fields_absent(self) -> None:
        record = _record()
        assert record.authors == []
        assert record.first_publish_year is None
        assert record.edition_count == 0
        assert record.ratings_average is None
        assert record.ratings_count == 0
        assert record.subjects == []
        assert record.cover_id is None

    def test_source_genres_multiple_values(self) -> None:
        record = _record(source_genres=["mystery", "post apocalyptic"])
        assert record.source_genres == ["mystery", "post apocalyptic"]

    def test_missing_work_key_raises(self) -> None:
        with pytest.raises(ValidationError):
            BookRecord(title="No Key", source_genres=["mystery"])  # type: ignore[call-arg]

    def test_missing_title_raises(self) -> None:
        with pytest.raises(ValidationError):
            BookRecord(work_key="/works/OL1W", source_genres=["mystery"])  # type: ignore[call-arg]

    def test_missing_source_genres_raises(self) -> None:
        with pytest.raises(ValidationError):
            BookRecord(work_key="/works/OL1W", title="Some Title")  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        record = _record()
        with pytest.raises(ValidationError):
            record.title = "Changed"  # type: ignore[misc]

    def test_ratings_average_none_is_valid(self) -> None:
        record = _record(ratings_average=None)
        assert record.ratings_average is None

    def test_cover_id_none_is_valid(self) -> None:
        record = _record(cover_id=None)
        assert record.cover_id is None

    def test_first_publish_year_none_is_valid(self) -> None:
        record = _record(first_publish_year=None)
        assert record.first_publish_year is None


# ---------------------------------------------------------------------------
# BookFetcherAgentInput
# ---------------------------------------------------------------------------

class TestBookFetcherAgentInput:
    def test_valid_single_genre(self) -> None:
        inp = BookFetcherAgentInput(normalized_genres=["mystery"])
        assert inp.normalized_genres == ["mystery"]
        assert inp.limit_per_genre == 30

    def test_valid_multiple_genres(self) -> None:
        inp = BookFetcherAgentInput(normalized_genres=["mystery", "post_apocalyptic"])
        assert len(inp.normalized_genres) == 2

    def test_custom_limit(self) -> None:
        inp = BookFetcherAgentInput(normalized_genres=["fantasy"], limit_per_genre=10)
        assert inp.limit_per_genre == 10

    def test_empty_genres_raises(self) -> None:
        with pytest.raises(ValidationError):
            BookFetcherAgentInput(normalized_genres=[])


# ---------------------------------------------------------------------------
# BookFetcherAgentOutput
# ---------------------------------------------------------------------------

class TestBookFetcherAgentOutput:
    def test_valid_with_books(self) -> None:
        books = [
            _record(source_genres=["mystery"]),
            _record(work_key="/works/OL2W", title="Other", source_genres=["fantasy"]),
        ]
        out = _output(books=books, queries_executed=["mystery", "fantasy"])
        assert len(out.books) == 2
        assert out.queries_executed == ["mystery", "fantasy"]

    def test_valid_empty_books(self) -> None:
        out = _output(books=[], queries_executed=["mystery"])
        assert out.books == []

    def test_debug_defaults_to_empty_dict(self) -> None:
        out = _output()
        assert out.debug == {}

    def test_debug_accepts_populated_dict(self) -> None:
        debug = {
            "per_genre": {"mystery": {"books_fetched": 10, "cache": "miss"}},
            "total_raw": 10,
            "total_after_dedup": 10,
            "duplicates_merged": 0,
        }
        out = _output(debug=debug)
        assert out.debug["total_raw"] == 10
        assert out.debug["duplicates_merged"] == 0

    def test_frozen(self) -> None:
        out = _output()
        with pytest.raises(ValidationError):
            out.books = []  # type: ignore[misc]

    def test_schema_version(self) -> None:
        out = _output()
        assert out.schema_version == BOOK_FETCHER_SCHEMA_VERSION

    def test_schema_version_value(self) -> None:
        assert BOOK_FETCHER_SCHEMA_VERSION == "1.1"


# ---------------------------------------------------------------------------
# Round-trip JSON validation
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_input_roundtrip(self) -> None:
        original = BookFetcherAgentInput(
            normalized_genres=["mystery", "post_apocalyptic"],
            limit_per_genre=15,
        )
        reconstructed = BookFetcherAgentInput.model_validate_json(
            original.model_dump_json()
        )
        assert reconstructed == original

    def test_output_roundtrip(self) -> None:
        original = _output(
            books=[_record(
                authors=["Cormac McCarthy"],
                source_genres=["post apocalyptic", "literary fiction"],
            )],
            queries_executed=["post apocalyptic", "literary fiction"],
            debug={
                "per_genre": {
                    "post apocalyptic": {"books_fetched": 1, "cache": "miss"},
                    "literary fiction": {"books_fetched": 1, "cache": "hit"},
                },
                "total_raw": 2,
                "total_after_dedup": 1,
                "duplicates_merged": 1,
            },
        )
        reconstructed = BookFetcherAgentOutput.model_validate_json(
            original.model_dump_json()
        )
        assert reconstructed == original
