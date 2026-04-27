"""Tests for the RunInspector class and HTML generation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import orjson
import pytest

from storymesh.core.artifacts import ArtifactStore
from storymesh.core.run_inspector import (
    RunInspection,
    RunInspector,
    RunMetadata,
    StageStatus,
)
from storymesh.exceptions import RunNotFoundError

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_SAMPLE_METADATA: dict[str, Any] = {
    "user_prompt": "a dark fantasy thriller",
    "pipeline_version": "0.6.0",
    "timestamp": "2026-04-03T12:00:00+00:00",
    "run_id": "run_abc",
    "stage_timings": {"genre_normalizer": 0.5, "book_fetcher": 1.2},
}

_GENRE_NORMALIZER_OUTPUT: dict[str, Any] = {
    "raw_input": "a dark fantasy thriller",
    "normalized_genres": ["fantasy", "thriller"],
    "subgenres": ["dark fantasy"],
    "user_tones": ["dark", "tense"],
    "tone_override": False,
    "override_note": None,
    "narrative_context": ["urban", "modern"],
    "inferred_genres": [{"canonical_genre": "horror", "confidence": 0.6}],
    "debug": {},
    "schema_version": "3.1",
}

_LLM_CALL_RECORD: dict[str, Any] = {
    "ts": "2026-04-03T12:00:01+00:00",
    "run_id": "run_abc",
    "agent": "genre_normalizer",
    "model": "claude-haiku-4-5-20251001",
    "temperature": 0.0,
    "attempt": 1,
    "system_prompt": "You are a genre classifier.\nBe precise.",
    "user_prompt": "Classify: a dark fantasy thriller",
    "raw_response": '{"genres": ["fantasy", "thriller"]}',
    "parse_success": True,
    "latency_ms": 906,
}


def _write_run(
    tmp_path: Path,
    run_id: str,
    metadata: dict[str, Any] | None = None,
    stages: dict[str, dict[str, Any]] | None = None,
    llm_records: list[dict[str, Any]] | None = None,
) -> Path:
    """Write a run directory with optional metadata, stage files, and LLM records."""
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if metadata is not None:
        (run_dir / "run_metadata.json").write_bytes(orjson.dumps(metadata))

    for stage_name, data in (stages or {}).items():
        (run_dir / f"{stage_name}_output.json").write_bytes(orjson.dumps(data))

    if llm_records:
        lines = b"\n".join(orjson.dumps(r) for r in llm_records) + b"\n"
        (run_dir / "llm_calls.jsonl").write_bytes(lines)

    return run_dir


def _make_inspector(tmp_path: Path) -> RunInspector:
    """Return a RunInspector backed by a fresh ArtifactStore in tmp_path."""
    store = ArtifactStore(root=tmp_path)
    return RunInspector(store)


# ---------------------------------------------------------------------------
# Group A — list_run_ids (via ArtifactStore, tested directly in test_artifacts;
#           these tests exercise the inspector's resolution behaviour)
# ---------------------------------------------------------------------------


def test_load_latest_resolves_to_newest_run(tmp_path: Path) -> None:
    """inspector.load('latest') returns the most recently modified run."""
    _write_run(tmp_path, "run_old", metadata=_SAMPLE_METADATA | {"run_id": "run_old"})
    time.sleep(0.05)
    _write_run(tmp_path, "run_new", metadata=_SAMPLE_METADATA | {"run_id": "run_new"})

    inspector = _make_inspector(tmp_path)
    report = inspector.load("latest")

    assert report.run_id == "run_new"


def test_load_latest_raises_when_no_runs_exist(tmp_path: Path) -> None:
    """inspector.load('latest') raises RunNotFoundError when there are no runs."""
    inspector = _make_inspector(tmp_path)

    with pytest.raises(RunNotFoundError):
        inspector.load("latest")


def test_load_raises_for_nonexistent_run_id(tmp_path: Path) -> None:
    """inspector.load('<id>') raises RunNotFoundError for an absent run directory."""
    inspector = _make_inspector(tmp_path)

    with pytest.raises(RunNotFoundError):
        inspector.load("does_not_exist")


# ---------------------------------------------------------------------------
# Group B — RunInspector.load()
# ---------------------------------------------------------------------------


def test_load_done_stage_when_artifact_present(tmp_path: Path) -> None:
    """A present, valid stage file is reported as DONE with its data."""
    _write_run(
        tmp_path,
        "run_abc",
        metadata=_SAMPLE_METADATA,
        stages={"genre_normalizer": _GENRE_NORMALIZER_OUTPUT},
    )
    inspector = _make_inspector(tmp_path)
    report = inspector.load("run_abc")

    stage = report.stages["genre_normalizer"]
    assert stage.status == StageStatus.DONE
    assert stage.data is not None
    assert stage.data["normalized_genres"] == ["fantasy", "thriller"]


def test_load_missing_stage_when_file_absent(tmp_path: Path) -> None:
    """An absent stage file is reported as MISSING with no data."""
    _write_run(tmp_path, "run_abc", metadata=_SAMPLE_METADATA)
    inspector = _make_inspector(tmp_path)
    report = inspector.load("run_abc")

    stage = report.stages["book_fetcher"]
    assert stage.status == StageStatus.MISSING
    assert stage.data is None


def test_load_corrupt_stage_when_json_invalid(tmp_path: Path) -> None:
    """A stage file containing invalid JSON is reported as CORRUPT."""
    run_dir = _write_run(tmp_path, "run_abc", metadata=_SAMPLE_METADATA)
    (run_dir / "book_ranker_output.json").write_bytes(b"not valid json {{{")

    inspector = _make_inspector(tmp_path)
    report = inspector.load("run_abc")

    stage = report.stages["book_ranker"]
    assert stage.status == StageStatus.CORRUPT
    assert stage.data is None


def test_load_corrupt_stage_when_not_a_dict(tmp_path: Path) -> None:
    """A stage file whose root value is not a JSON object is reported as CORRUPT."""
    run_dir = _write_run(tmp_path, "run_abc", metadata=_SAMPLE_METADATA)
    (run_dir / "genre_normalizer_output.json").write_bytes(orjson.dumps([1, 2, 3]))

    inspector = _make_inspector(tmp_path)
    report = inspector.load("run_abc")

    assert report.stages["genre_normalizer"].status == StageStatus.CORRUPT


def test_load_metadata_none_when_file_absent(tmp_path: Path) -> None:
    """metadata is None when run_metadata.json does not exist."""
    _write_run(tmp_path, "run_abc")  # no metadata kwarg

    inspector = _make_inspector(tmp_path)
    report = inspector.load("run_abc")

    assert report.metadata is None


def test_load_metadata_populated_when_file_present(tmp_path: Path) -> None:
    """Metadata fields are correctly loaded from run_metadata.json."""
    _write_run(tmp_path, "run_abc", metadata=_SAMPLE_METADATA)

    inspector = _make_inspector(tmp_path)
    report = inspector.load("run_abc")

    assert report.metadata is not None
    assert isinstance(report.metadata, RunMetadata)
    assert report.metadata.user_prompt == "a dark fantasy thriller"
    assert report.metadata.stage_timings["genre_normalizer"] == 0.5


def test_load_metadata_stage_timings_empty_when_absent(tmp_path: Path) -> None:
    """stage_timings defaults to an empty dict when key is missing from metadata."""
    meta = {k: v for k, v in _SAMPLE_METADATA.items() if k != "stage_timings"}
    _write_run(tmp_path, "run_abc", metadata=meta)

    inspector = _make_inspector(tmp_path)
    report = inspector.load("run_abc")

    assert report.metadata is not None
    assert report.metadata.stage_timings == {}


def test_load_llm_calls_populated(tmp_path: Path) -> None:
    """LLM call records are loaded from llm_calls.jsonl."""
    _write_run(tmp_path, "run_abc", metadata=_SAMPLE_METADATA, llm_records=[_LLM_CALL_RECORD])

    inspector = _make_inspector(tmp_path)
    report = inspector.load("run_abc")

    assert len(report.llm_calls) == 1
    call = report.llm_calls[0]
    assert call.agent == "genre_normalizer"
    assert call.parse_success is True
    assert call.latency_ms == 906.0
    assert "classifier" in call.system_prompt


def test_load_llm_calls_empty_when_file_absent(tmp_path: Path) -> None:
    """llm_calls is an empty list when llm_calls.jsonl does not exist."""
    _write_run(tmp_path, "run_abc", metadata=_SAMPLE_METADATA)

    inspector = _make_inspector(tmp_path)
    report = inspector.load("run_abc")

    assert report.llm_calls == []


def test_load_llm_calls_skips_corrupt_line(tmp_path: Path) -> None:
    """A corrupt JSONL line is skipped; remaining records are still loaded."""
    run_dir = _write_run(tmp_path, "run_abc", metadata=_SAMPLE_METADATA)
    (run_dir / "llm_calls.jsonl").write_bytes(
        b"not valid json\n" + orjson.dumps(_LLM_CALL_RECORD) + b"\n"
    )

    inspector = _make_inspector(tmp_path)
    report = inspector.load("run_abc")

    assert len(report.llm_calls) == 1
    assert report.llm_calls[0].agent == "genre_normalizer"


# ---------------------------------------------------------------------------
# Group C — generate_html()
# ---------------------------------------------------------------------------


def _minimal_report(tmp_path: Path) -> RunInspection:
    """Load a minimal run inspection for HTML generation tests."""
    _write_run(
        tmp_path,
        "run_abc",
        metadata=_SAMPLE_METADATA,
        stages={"genre_normalizer": _GENRE_NORMALIZER_OUTPUT},
        llm_records=[_LLM_CALL_RECORD],
    )
    return _make_inspector(tmp_path).load("run_abc")


def test_generate_html_contains_doctype(tmp_path: Path) -> None:
    """generate_html() output starts with a DOCTYPE declaration."""
    report = _minimal_report(tmp_path)
    html = _make_inspector(tmp_path).generate_html(report)

    assert "<!DOCTYPE html>" in html


def test_generate_html_contains_run_id(tmp_path: Path) -> None:
    """generate_html() embeds the run ID in the output."""
    report = _minimal_report(tmp_path)
    html = _make_inspector(tmp_path).generate_html(report)

    assert "run_abc" in html


def test_generate_html_escapes_user_prompt(tmp_path: Path) -> None:
    """generate_html() HTML-escapes user content to prevent injection."""
    dangerous_meta = _SAMPLE_METADATA | {"user_prompt": "<script>alert(1)</script>"}
    _write_run(tmp_path, "run_xss", metadata=dangerous_meta)
    report = _make_inspector(tmp_path).load("run_xss")
    html = _make_inspector(tmp_path).generate_html(report)

    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_generate_html_contains_pre_for_llm_prompts(tmp_path: Path) -> None:
    """generate_html() renders LLM prompts inside <pre> elements."""
    report = _minimal_report(tmp_path)
    html = _make_inspector(tmp_path).generate_html(report)

    assert "<pre>" in html


def test_generate_html_missing_stage_renders_without_error(tmp_path: Path) -> None:
    """generate_html() renders cleanly when stages are MISSING."""
    _write_run(tmp_path, "run_nometa")  # no stages, no metadata

    report = _make_inspector(tmp_path).load("run_nometa")
    # Should not raise
    html = _make_inspector(tmp_path).generate_html(report)

    assert "<!DOCTYPE html>" in html
    assert "Stage not yet run" in html


# ---------------------------------------------------------------------------
# Group D — stage-specific HTML body renderers
# ---------------------------------------------------------------------------


_BOOK_FETCHER_OUTPUT: dict[str, Any] = {
    "books": [
        {
            "work_key": "/works/OL1W",
            "title": "The Drowned Archive",
            "authors": ["Alice Rivers", "Jun Tanaka"],
            "first_publish_year": 2021,
            "source_genres": ["mystery"],
        },
    ],
    "queries_executed": ["mystery", "post apocalyptic"],
}

_BOOK_RANKER_OUTPUT: dict[str, Any] = {
    "ranked_books": [
        {
            "book": {
                "work_key": "/works/OL1W",
                "title": "The Drowned Archive",
                "authors": ["Alice Rivers"],
                "source_genres": ["mystery"],
            },
            "rank": 1,
            "composite_score": 0.873,
            "score_breakdown": {
                "genre_overlap": 0.9,
                "reader_engagement": 0.75,
            },
        },
    ],
    "ranked_summaries": [],
    "dropped_count": 2,
    "llm_reranked": False,
    "debug": {},
}

_THEME_EXTRACTOR_OUTPUT: dict[str, Any] = {
    "genre_clusters": [
        {
            "genre": "mystery",
            "books": ["The Drowned Archive"],
            "thematic_assumptions": ["Truth is recoverable"],
            "dominant_tropes": [],
        }
    ],
    "tensions": [
        {
            "tension_id": "T1",
            "cluster_a": "mystery",
            "assumption_a": "Truth is recoverable",
            "cluster_b": "post_apocalyptic",
            "assumption_b": "Records no longer exist",
            "creative_question": "What does investigation mean without infrastructure?",
            "intensity": 0.9,
            "cliched_resolutions": [],
        }
    ],
    "narrative_seeds": [
        {
            "seed_id": "S1",
            "concept": "A scavenger detective reinvents investigation.",
            "tensions_used": ["T1"],
            "tonal_direction": [],
            "narrative_context_used": [],
        }
    ],
    "user_tones_carried": ["bleak"],
}

_PROPOSAL_DRAFT_OUTPUT: dict[str, Any] = {
    "proposal": {
        "seed_id": "S1",
        "title": "The Last Inquest",
        "protagonist": "Mara Voss, a former detective.",
        "setting": "A flooded city-state.",
        "plot_arc": "Three-act reconstruction.",
        "thematic_thesis": "Justice requires no institution.",
        "key_scenes": ["Mara finds the body.", "Tribunal convenes."],
        "tensions_addressed": ["T1"],
        "tone": ["bleak"],
        "genre_blend": ["mystery", "post_apocalyptic"],
    },
    "selection_rationale": {
        "selected_index": 1,
        "rationale": "Candidate 1 addressed the thematic question most directly.",
        # See note in test_cli.py: the renderer passes this through _list, so
        # supply a list here rather than the schema-typed dict[str, list[str]].
        "cliche_violations": ["leans on the lone-savior trope"],
        "runner_up_index": 2,
    },
    "debug": {
        "num_candidates_requested": 3,
        "num_valid_candidates": 2,
        "num_parse_failures": 1,
        "draft_temperature": 0.9,
        "selection_temperature": 0.2,
        "total_llm_calls": 4,
    },
}


def test_generate_html_renders_book_fetcher_stage(tmp_path: Path) -> None:
    """book_fetcher HTML body embeds queries, book title, authors, and year."""
    _write_run(
        tmp_path,
        "run_bf",
        metadata=_SAMPLE_METADATA,
        stages={"book_fetcher": _BOOK_FETCHER_OUTPUT},
    )
    report = _make_inspector(tmp_path).load("run_bf")
    html_out = _make_inspector(tmp_path).generate_html(report)

    assert "The Drowned Archive" in html_out
    assert "Alice Rivers" in html_out
    assert "2021" in html_out
    # Queries list should contain both executed subjects.
    assert "post apocalyptic" in html_out


def test_generate_html_renders_book_ranker_stage(tmp_path: Path) -> None:
    """book_ranker HTML body embeds rank, title, score, and breakdown."""
    _write_run(
        tmp_path,
        "run_br",
        metadata=_SAMPLE_METADATA,
        stages={"book_ranker": _BOOK_RANKER_OUTPUT},
    )
    report = _make_inspector(tmp_path).load("run_br")
    html_out = _make_inspector(tmp_path).generate_html(report)

    assert "The Drowned Archive" in html_out
    # Composite score renders with three decimal places.
    assert "0.873" in html_out
    assert "0.900" in html_out  # genre_overlap
    assert "0.750" in html_out  # reader_engagement


def test_generate_html_renders_theme_extractor_stage(tmp_path: Path) -> None:
    """theme_extractor HTML body embeds clusters, tensions, and seeds."""
    _write_run(
        tmp_path,
        "run_te",
        metadata=_SAMPLE_METADATA,
        stages={"theme_extractor": _THEME_EXTRACTOR_OUTPUT},
    )
    report = _make_inspector(tmp_path).load("run_te")
    html_out = _make_inspector(tmp_path).generate_html(report)

    assert "Genre Clusters" in html_out
    assert "Thematic Tensions" in html_out
    assert "Narrative Seeds" in html_out
    # IDs render in their sections.
    assert "T1" in html_out
    assert "S1" in html_out
    # Intensity is formatted to two decimal places.
    assert "0.90" in html_out


def test_generate_html_renders_proposal_draft_stage(tmp_path: Path) -> None:
    """proposal_draft HTML body embeds title, protagonist, rationale, and cliché block."""
    _write_run(
        tmp_path,
        "run_pd",
        metadata=_SAMPLE_METADATA,
        stages={"proposal_draft": _PROPOSAL_DRAFT_OUTPUT},
    )
    report = _make_inspector(tmp_path).load("run_pd")
    html_out = _make_inspector(tmp_path).generate_html(report)

    assert "The Last Inquest" in html_out
    assert "Mara Voss" in html_out
    assert "Key Scenes" in html_out
    # Runner-up reference surfaces in the selection block.
    assert "runner-up: 2" in html_out
    # Cliché violations block renders because we passed a non-empty list.
    assert "Cliché Violations" in html_out


def test_generate_html_renders_corrupt_stage(tmp_path: Path) -> None:
    """A CORRUPT stage renders with the 'could not be parsed' fallback body."""
    run_dir = _write_run(tmp_path, "run_corrupt", metadata=_SAMPLE_METADATA)
    # Write an invalid JSON root (a list) — loader reports CORRUPT.
    (run_dir / "proposal_draft_output.json").write_bytes(orjson.dumps([1, 2, 3]))

    report = _make_inspector(tmp_path).load("run_corrupt")
    assert report.stages["proposal_draft"].status == StageStatus.CORRUPT
    html_out = _make_inspector(tmp_path).generate_html(report)

    assert "could not be parsed" in html_out


def test_generate_html_renders_rubric_judge_fallback(tmp_path: Path) -> None:
    """The generic ``<pre>`` fallback renders for stages without a dedicated body."""
    _write_run(
        tmp_path,
        "run_rubric",
        metadata=_SAMPLE_METADATA,
        stages={"rubric_judge": {"passed": True, "score": 0.9}},
    )
    report = _make_inspector(tmp_path).load("run_rubric")
    html_out = _make_inspector(tmp_path).generate_html(report)

    # The generic fallback escapes the dict and wraps it in <pre>.
    assert "<pre>" in html_out
    assert "passed" in html_out


# ---------------------------------------------------------------------------
# Group E — internal helpers
# ---------------------------------------------------------------------------


def test_as_float_returns_none_for_none() -> None:
    """_as_float(None) returns None rather than raising."""
    from storymesh.core.run_inspector import _as_float

    assert _as_float(None) is None


def test_as_float_returns_none_for_non_numeric() -> None:
    """_as_float on a non-numeric string returns None."""
    from storymesh.core.run_inspector import _as_float

    assert _as_float("not a number") is None


def test_as_float_converts_valid_numeric_string() -> None:
    """_as_float('3.14') returns 3.14 as a float."""
    from storymesh.core.run_inspector import _as_float

    assert _as_float("3.14") == 3.14


def test_load_llm_calls_handles_missing_latency_and_temperature(tmp_path: Path) -> None:
    """LLM records with missing latency/temperature load with those fields as None."""
    record = {
        k: v
        for k, v in _LLM_CALL_RECORD.items()
        if k not in {"latency_ms", "temperature"}
    }
    _write_run(tmp_path, "run_sparse", metadata=_SAMPLE_METADATA, llm_records=[record])

    report = _make_inspector(tmp_path).load("run_sparse")
    assert len(report.llm_calls) == 1
    assert report.llm_calls[0].latency_ms is None
    assert report.llm_calls[0].temperature is None


def test_load_llm_calls_skips_non_dict_lines(tmp_path: Path) -> None:
    """A JSONL line whose root is not a dict is skipped with a warning."""
    run_dir = _write_run(tmp_path, "run_notdict", metadata=_SAMPLE_METADATA)
    (run_dir / "llm_calls.jsonl").write_bytes(
        orjson.dumps([1, 2, 3]) + b"\n" + orjson.dumps(_LLM_CALL_RECORD) + b"\n"
    )

    report = _make_inspector(tmp_path).load("run_notdict")
    assert len(report.llm_calls) == 1
    assert report.llm_calls[0].agent == "genre_normalizer"


def test_load_metadata_non_dict_returns_none(tmp_path: Path) -> None:
    """run_metadata.json whose root is a list (not a dict) is rejected as None."""
    run_dir = _write_run(tmp_path, "run_meta_bad")
    (run_dir / "run_metadata.json").write_bytes(orjson.dumps([1, 2, 3]))

    report = _make_inspector(tmp_path).load("run_meta_bad")
    assert report.metadata is None


def test_load_metadata_invalid_json_returns_none(tmp_path: Path) -> None:
    """Unparseable run_metadata.json is treated as missing rather than raising."""
    run_dir = _write_run(tmp_path, "run_meta_corrupt")
    (run_dir / "run_metadata.json").write_bytes(b"not valid json {{{")

    report = _make_inspector(tmp_path).load("run_meta_corrupt")
    assert report.metadata is None


def test_load_metadata_stage_timings_non_dict_ignored(tmp_path: Path) -> None:
    """stage_timings that is not a dict falls back to {} instead of raising."""
    bad_meta = dict(_SAMPLE_METADATA)
    bad_meta["stage_timings"] = "not a dict"  # type: ignore[assignment]
    _write_run(tmp_path, "run_st_bad", metadata=bad_meta)

    report = _make_inspector(tmp_path).load("run_st_bad")
    assert report.metadata is not None
    assert report.metadata.stage_timings == {}
