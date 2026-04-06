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
