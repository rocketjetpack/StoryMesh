from pathlib import Path

import orjson

from storymesh.core.artifacts import ArtifactStore


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