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