from pathlib import Path
from typing import Any

import orjson
from pydantic import BaseModel

from storymesh.core.artifacts import ArtifactStore, persist_node_output


def test_artifact_save_load(tmp_path: Path) -> None:
    store = ArtifactStore(root=tmp_path)
    stage_name = "genre_normalizer"
    stage_hash = "abc123"
    data = {"foo": "bar", "number": 42, "fizz": "buzz"}

    store.save_stage(stage_name, stage_hash, data)
    loaded = store.load_stage(stage_name, stage_hash)

    assert loaded == data

def test_missing_stage_returns_none(tmp_path: Path) -> None:
    store = ArtifactStore(root=tmp_path)
    loaded = store.load_stage("genre_normalizer", "nonexistent_hash")
    assert loaded is None

def test_stage_file_written_where_expected(tmp_path: Path) -> None:
    store = ArtifactStore(root=tmp_path)
    stage_name = "this_is_not_a_real_stage"
    stage_hash = "stagdance"
    data = {"foo": "bar", "number": 42, "fizz": "buzz"}

    store.save_stage(stage_name, stage_hash, data)

    expected_path = (tmp_path / "stages" / stage_name / f"{stage_hash}.json")
    assert expected_path.exists()

def test_save_run_creates_run_dir(tmp_path: Path) -> None:
    store = ArtifactStore(root=tmp_path)

    run_hash = "run123"
    metadata = {"input": "fantasy"}

    store.save_run(run_hash, metadata)

    expected_file = (
        tmp_path
        / "runs"
        / run_hash
        / "run_metadata.json"
    )

    assert expected_file.exists()

def test_saved_run_metadata_is_valid_json(tmp_path: Path) -> None:
    store = ArtifactStore(root=tmp_path)

    run_hash = "run456"
    metadata = {"input": "near-future sci-fi", "parameters": {"temperature": 0.7}}
    store.save_run(run_hash, metadata)

    file_path = (
        tmp_path / "runs" / run_hash / "run_metadata.json"
    )

    raw = file_path.read_bytes()
    parsed = orjson.loads(raw)

    assert parsed == metadata


# ---------------------------------------------------------------------------
# persist_node_output
# ---------------------------------------------------------------------------


class _FakeOutput(BaseModel):
    """Minimal Pydantic model for testing persist_node_output."""

    value: str
    count: int = 0


def test_persist_node_output_pydantic_model(tmp_path: Path) -> None:
    """Pydantic model output is serialised via model_dump and written to disk."""
    store = ArtifactStore(root=tmp_path)
    output = _FakeOutput(value="hello", count=42)

    persist_node_output(store, "run_abc", "test_stage", output)

    expected = tmp_path / "runs" / "run_abc" / "test_stage_output.json"
    assert expected.exists()
    parsed = orjson.loads(expected.read_bytes())
    assert parsed == {"value": "hello", "count": 42}


def test_persist_node_output_dict(tmp_path: Path) -> None:
    """Plain dict output is written directly to disk."""
    store = ArtifactStore(root=tmp_path)
    output: dict[str, Any] = {"key": "val", "num": 7}

    persist_node_output(store, "run_def", "dict_stage", output)

    expected = tmp_path / "runs" / "run_def" / "dict_stage_output.json"
    assert expected.exists()
    parsed = orjson.loads(expected.read_bytes())
    assert parsed == output


def test_persist_node_output_none_is_noop(tmp_path: Path) -> None:
    """None output must not create any files."""
    store = ArtifactStore(root=tmp_path)

    persist_node_output(store, "run_ghi", "noop_stage", None)

    run_dir = tmp_path / "runs" / "run_ghi"
    assert not run_dir.exists()


def test_persist_node_output_unserialisable_is_noop(tmp_path: Path) -> None:
    """Output that is neither a Pydantic model nor a dict is silently skipped."""
    store = ArtifactStore(root=tmp_path)

    persist_node_output(store, "run_jkl", "bad_stage", 42)

    run_dir = tmp_path / "runs" / "run_jkl"
    assert not run_dir.exists()


def test_persist_node_output_filename(tmp_path: Path) -> None:
    """Output filename follows the ``<stage_name>_output.json`` convention."""
    store = ArtifactStore(root=tmp_path)
    output = _FakeOutput(value="check")

    persist_node_output(store, "run_mno", "genre_normalizer", output)

    assert (tmp_path / "runs" / "run_mno" / "genre_normalizer_output.json").exists()


# ---------------------------------------------------------------------------
# ArtifactStore.log_llm_call
# ---------------------------------------------------------------------------


def test_log_llm_call_creates_jsonl_file(tmp_path: Path) -> None:
    """log_llm_call() creates llm_calls.jsonl in the correct run directory."""
    store = ArtifactStore(root=tmp_path)
    record: dict[str, Any] = {"agent": "theme_extractor", "parse_success": True}

    store.log_llm_call("run_abc", record)

    assert (tmp_path / "runs" / "run_abc" / "llm_calls.jsonl").exists()


def test_log_llm_call_two_calls_appends_two_lines(tmp_path: Path) -> None:
    """Calling log_llm_call() twice produces exactly two JSONL lines."""
    store = ArtifactStore(root=tmp_path)
    store.log_llm_call("run_abc", {"seq": 1})
    store.log_llm_call("run_abc", {"seq": 2})

    lines = (tmp_path / "runs" / "run_abc" / "llm_calls.jsonl").read_bytes().splitlines()
    assert len(lines) == 2


def test_log_llm_call_lines_are_valid_json(tmp_path: Path) -> None:
    """Each line in llm_calls.jsonl round-trips through orjson.loads()."""
    store = ArtifactStore(root=tmp_path)
    store.log_llm_call("run_abc", {"agent": "genre_normalizer", "parse_success": True})
    store.log_llm_call("run_abc", {"agent": "theme_extractor", "parse_success": False})

    lines = (tmp_path / "runs" / "run_abc" / "llm_calls.jsonl").read_bytes().splitlines()
    records = [orjson.loads(line) for line in lines]
    assert records[0]["agent"] == "genre_normalizer"
    assert records[1]["agent"] == "theme_extractor"


# ---------------------------------------------------------------------------
# ArtifactStore.purge_stage_cache
# ---------------------------------------------------------------------------


def test_purge_stage_cache_removes_files_and_returns_count(tmp_path: Path) -> None:
    """purge_stage_cache() deletes all cached files and returns the correct count."""
    store = ArtifactStore(root=tmp_path)
    store.save_stage("genre_normalizer", "hash1", {"a": 1})
    store.save_stage("genre_normalizer", "hash2", {"b": 2})
    store.save_stage("book_fetcher", "hash3", {"c": 3})

    count = store.purge_stage_cache()

    assert count == 3
    assert not list((tmp_path / "stages").rglob("*.json"))


def test_purge_stage_cache_stage_subdirs_removed(tmp_path: Path) -> None:
    """purge_stage_cache() removes per-stage subdirectories, not just their contents."""
    store = ArtifactStore(root=tmp_path)
    store.save_stage("genre_normalizer", "abc", {"x": 1})

    store.purge_stage_cache()

    assert not (tmp_path / "stages" / "genre_normalizer").exists()


def test_purge_stage_cache_empty_store_returns_zero(tmp_path: Path) -> None:
    """purge_stage_cache() returns 0 when there are no cached files."""
    store = ArtifactStore(root=tmp_path)

    assert store.purge_stage_cache() == 0


def test_purge_stage_cache_nonexistent_stages_dir_returns_zero(tmp_path: Path) -> None:
    """purge_stage_cache() handles a missing stages directory without error."""
    store = ArtifactStore(root=tmp_path)
    store.stages_dir.rmdir()

    assert store.purge_stage_cache() == 0


# ---------------------------------------------------------------------------
# ArtifactStore.purge_runs
# ---------------------------------------------------------------------------


def test_purge_runs_removes_run_dirs_and_returns_count(tmp_path: Path) -> None:
    """purge_runs() deletes all run directories and returns the correct count."""
    store = ArtifactStore(root=tmp_path)
    store.save_run("run1", {"prompt": "a"})
    store.save_run("run2", {"prompt": "b"})

    count = store.purge_runs()

    assert count == 2
    assert not list((tmp_path / "runs").iterdir())


def test_purge_runs_empty_store_returns_zero(tmp_path: Path) -> None:
    """purge_runs() returns 0 when there are no run directories."""
    store = ArtifactStore(root=tmp_path)

    assert store.purge_runs() == 0


def test_purge_runs_nonexistent_runs_dir_returns_zero(tmp_path: Path) -> None:
    """purge_runs() handles a missing runs directory without error."""
    store = ArtifactStore(root=tmp_path)
    store.runs_dir.rmdir()

    assert store.purge_runs() == 0


def test_purge_runs_preserves_runs_dir_itself(tmp_path: Path) -> None:
    """purge_runs() leaves the parent runs directory in place."""
    store = ArtifactStore(root=tmp_path)
    store.save_run("run1", {"prompt": "x"})

    store.purge_runs()

    assert store.runs_dir.exists()


def test_log_llm_call_creates_run_dir_if_absent(tmp_path: Path) -> None:
    """log_llm_call() creates the run directory even without a run_metadata.json."""
    store = ArtifactStore(root=tmp_path)
    run_dir = tmp_path / "runs" / "brand_new_run"
    assert not run_dir.exists()

    store.log_llm_call("brand_new_run", {"agent": "test"})

    assert run_dir.exists()
    assert (run_dir / "llm_calls.jsonl").exists()


# ---------------------------------------------------------------------------
# ArtifactStore.load_run_file
# ---------------------------------------------------------------------------


def test_load_run_file_returns_bytes_for_existing_file(tmp_path: Path) -> None:
    """load_run_file() returns the raw bytes of an existing file."""
    store = ArtifactStore(root=tmp_path)
    store.save_run("run1", {"x": 1})

    result = store.load_run_file("run1", "run_metadata.json")

    assert result is not None
    assert b"run1" in result or b'"x"' in result


def test_load_run_file_returns_none_for_absent_file(tmp_path: Path) -> None:
    """load_run_file() returns None when the file does not exist."""
    store = ArtifactStore(root=tmp_path)

    assert store.load_run_file("nonexistent_run", "run_metadata.json") is None


# ---------------------------------------------------------------------------
# ArtifactStore.list_run_ids
# ---------------------------------------------------------------------------


def test_list_run_ids_empty_when_no_runs(tmp_path: Path) -> None:
    """list_run_ids() returns an empty list when runs_dir has no subdirectories."""
    store = ArtifactStore(root=tmp_path)

    assert store.list_run_ids() == []


def test_list_run_ids_returns_newest_first(tmp_path: Path) -> None:
    """list_run_ids() returns run IDs sorted newest-first by mtime."""
    import time

    store = ArtifactStore(root=tmp_path)
    store.save_run("run_old", {"x": 1})
    time.sleep(0.05)
    store.save_run("run_new", {"x": 2})

    ids = store.list_run_ids()

    assert ids[0] == "run_new"
    assert ids[1] == "run_old"


def test_list_run_ids_handles_absent_runs_dir(tmp_path: Path) -> None:
    """list_run_ids() returns an empty list when runs_dir does not exist."""
    store = ArtifactStore(root=tmp_path)
    store.runs_dir.rmdir()

    assert store.list_run_ids() == []