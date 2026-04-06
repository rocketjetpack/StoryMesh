from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from storymesh.cli import app
from storymesh.versioning import __version__ as storymesh_version

runner = CliRunner()


def test_show_version() -> None:  # noqa: ANN201
    result = runner.invoke(app, ["show-version"])
    assert result.exit_code == 0
    assert storymesh_version in result.output
    assert "Schema Versions:" in result.output
    assert "Genre Constraint" in result.output
    assert "Agent Versions:" in result.output
    assert "Genre Normalizer" in result.output


# ---------------------------------------------------------------------------
# purge-cache
# ---------------------------------------------------------------------------


def test_purge_cache_both_with_yes_flag(tmp_path: Path) -> None:
    """purge-cache --yes clears both stage and API caches without prompting."""
    fake_store = MagicMock()
    fake_store.stages_dir = tmp_path / "stages"
    fake_store.purge_stage_cache.return_value = 3

    api_cache = tmp_path / "api_cache"
    api_cache.mkdir()

    with (
        patch("storymesh.cli.ArtifactStore", return_value=fake_store),
        patch(
            "storymesh.cli.get_config",
            return_value={"cache": {"dir": str(api_cache)}},
        ),
    ):
        result = runner.invoke(app, ["purge-cache", "--yes"])

    assert result.exit_code == 0
    fake_store.purge_stage_cache.assert_called_once()
    assert not api_cache.exists()
    assert "3 file(s) removed" in result.output


def test_purge_cache_stages_only(tmp_path: Path) -> None:
    """purge-cache --stages-only skips the API cache."""
    fake_store = MagicMock()
    fake_store.stages_dir = tmp_path / "stages"
    fake_store.purge_stage_cache.return_value = 1

    api_cache = tmp_path / "api_cache"
    api_cache.mkdir()

    with (
        patch("storymesh.cli.ArtifactStore", return_value=fake_store),
        patch(
            "storymesh.cli.get_config",
            return_value={"cache": {"dir": str(api_cache)}},
        ),
    ):
        result = runner.invoke(app, ["purge-cache", "--stages-only", "--yes"])

    assert result.exit_code == 0
    fake_store.purge_stage_cache.assert_called_once()
    assert api_cache.exists()  # untouched


def test_purge_cache_api_only(tmp_path: Path) -> None:
    """purge-cache --api-only skips the stage cache."""
    fake_store = MagicMock()
    fake_store.stages_dir = tmp_path / "stages"

    api_cache = tmp_path / "api_cache"
    api_cache.mkdir()

    with (
        patch("storymesh.cli.ArtifactStore", return_value=fake_store),
        patch(
            "storymesh.cli.get_config",
            return_value={"cache": {"dir": str(api_cache)}},
        ),
    ):
        result = runner.invoke(app, ["purge-cache", "--api-only", "--yes"])

    assert result.exit_code == 0
    fake_store.purge_stage_cache.assert_not_called()
    assert not api_cache.exists()


def test_purge_cache_api_cache_missing_is_graceful(tmp_path: Path) -> None:
    """purge-cache reports nothing to remove when API cache dir is absent."""
    fake_store = MagicMock()
    fake_store.stages_dir = tmp_path / "stages"
    fake_store.purge_stage_cache.return_value = 0

    with (
        patch("storymesh.cli.ArtifactStore", return_value=fake_store),
        patch(
            "storymesh.cli.get_config",
            return_value={"cache": {"dir": str(tmp_path / "nonexistent")}},
        ),
    ):
        result = runner.invoke(app, ["purge-cache", "--yes"])

    assert result.exit_code == 0
    assert "does not exist" in result.output


def test_purge_cache_aborts_on_no_confirmation() -> None:
    """purge-cache exits with non-zero when the user declines confirmation."""
    fake_store = MagicMock()
    fake_store.stages_dir = Path("/fake/stages")

    with (
        patch("storymesh.cli.ArtifactStore", return_value=fake_store),
        patch("storymesh.cli.get_config", return_value={}),
    ):
        result = runner.invoke(app, ["purge-cache"], input="n\n")

    assert result.exit_code != 0
    fake_store.purge_stage_cache.assert_not_called()


# ---------------------------------------------------------------------------
# purge-runs
# ---------------------------------------------------------------------------


def test_purge_runs_with_yes_flag() -> None:
    """purge-runs --yes removes run directories without prompting."""
    fake_store = MagicMock()
    fake_store.runs_dir = Path("/fake/runs")
    fake_store.purge_runs.return_value = 5

    with patch("storymesh.cli.ArtifactStore", return_value=fake_store):
        result = runner.invoke(app, ["purge-runs", "--yes"])

    assert result.exit_code == 0
    fake_store.purge_runs.assert_called_once()
    assert "5 run(s) removed" in result.output


def test_purge_runs_aborts_on_no_confirmation() -> None:
    """purge-runs exits with non-zero when the user declines confirmation."""
    fake_store = MagicMock()
    fake_store.runs_dir = Path("/fake/runs")

    with patch("storymesh.cli.ArtifactStore", return_value=fake_store):
        result = runner.invoke(app, ["purge-runs"], input="n\n")

    assert result.exit_code != 0
    fake_store.purge_runs.assert_not_called()


# ---------------------------------------------------------------------------
# inspect-run
# ---------------------------------------------------------------------------


def _write_cli_run(tmp_path: Path, run_id: str) -> None:
    """Write a minimal run directory for CLI inspect-run tests."""
    import orjson

    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "user_prompt": "a dark fantasy",
        "pipeline_version": "0.6.0",
        "timestamp": "2026-04-03T12:00:00+00:00",
        "run_id": run_id,
        "stage_timings": {"genre_normalizer": 0.5},
    }
    (run_dir / "run_metadata.json").write_bytes(orjson.dumps(meta))
    stage_data = {
        "normalized_genres": ["fantasy"],
        "subgenres": [],
        "user_tones": ["dark"],
        "tone_override": False,
        "override_note": None,
        "narrative_context": [],
        "inferred_genres": [],
        "debug": {},
        "schema_version": "3.1",
    }
    (run_dir / "genre_normalizer_output.json").write_bytes(orjson.dumps(stage_data))
    llm = {
        "ts": "2026-04-03T12:00:01+00:00",
        "run_id": run_id,
        "agent": "genre_normalizer",
        "model": "claude-haiku-4-5-20251001",
        "temperature": 0.0,
        "attempt": 1,
        "system_prompt": "You are a helper.\nLine two.",
        "user_prompt": "Classify: dark fantasy",
        "raw_response": '{"genres": ["fantasy"]}',
        "parse_success": True,
        "latency_ms": 500,
    }
    (run_dir / "llm_calls.jsonl").write_bytes(orjson.dumps(llm) + b"\n")


def test_inspect_run_valid_id_exits_zero(tmp_path: Path) -> None:
    """inspect-run with a valid run ID exits 0 and contains stage names."""
    _write_cli_run(tmp_path, "run_test")
    fake_store = MagicMock(wraps=__import__(
        "storymesh.core.artifacts", fromlist=["ArtifactStore"]).ArtifactStore(root=tmp_path)
        )

    with patch("storymesh.cli.ArtifactStore", return_value=fake_store):
        result = runner.invoke(app, ["inspect-run", "run_test"])

    assert result.exit_code == 0
    assert "genre_normalizer" in result.output


def test_inspect_run_no_arg_picks_latest(tmp_path: Path) -> None:
    """inspect-run with no argument picks the most recent run."""
    _write_cli_run(tmp_path, "only_run")
    from storymesh.core.artifacts import ArtifactStore as RealStore

    real_store = RealStore(root=tmp_path)

    with patch("storymesh.cli.ArtifactStore", return_value=real_store):
        result = runner.invoke(app, ["inspect-run"])

    assert result.exit_code == 0
    assert "only_run" in result.output


def test_inspect_run_nonexistent_id_exits_nonzero(tmp_path: Path) -> None:
    """inspect-run with a non-existent run ID exits non-zero with an error."""
    from storymesh.core.artifacts import ArtifactStore as RealStore

    real_store = RealStore(root=tmp_path)

    with patch("storymesh.cli.ArtifactStore", return_value=real_store):
        result = runner.invoke(app, ["inspect-run", "does_not_exist"])

    assert result.exit_code != 0


def test_inspect_run_llm_flag_shows_system_prompt(tmp_path: Path) -> None:
    """inspect-run --llm <agent> includes system prompt in output."""
    _write_cli_run(tmp_path, "run_llm")
    from storymesh.core.artifacts import ArtifactStore as RealStore

    real_store = RealStore(root=tmp_path)

    with patch("storymesh.cli.ArtifactStore", return_value=real_store):
        result = runner.invoke(app, ["inspect-run", "run_llm", "--llm", "genre_normalizer"])

    assert result.exit_code == 0
    assert "system prompt" in result.output


def test_inspect_run_html_flag_creates_file(tmp_path: Path) -> None:
    """inspect-run --html <path> creates the HTML report file."""
    _write_cli_run(tmp_path, "run_html")
    from storymesh.core.artifacts import ArtifactStore as RealStore

    real_store = RealStore(root=tmp_path)
    html_out = tmp_path / "report.html"

    with patch("storymesh.cli.ArtifactStore", return_value=real_store):
        result = runner.invoke(app, ["inspect-run", "run_html", "--html", str(html_out)])

    assert result.exit_code == 0
    assert html_out.exists()
    content = html_out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content


def test_generate_outputs_synopsis() -> None:  # noqa: ANN201
    mock_result = MagicMock()
    mock_result.final_synopsis = "A hero rises."
    mock_result.metadata = {
        "user_prompt": "fantasy",
        "run_id": "testrun123",
        "pipeline_version": "0.5.0",
        "stage_timings": {"genre_normalizer": 0.03, "book_fetcher": 1.24},
        "run_dir": "",
    }

    with patch("storymesh.cli.generate_synopsis", return_value=mock_result):
        result = runner.invoke(app, ["generate", "fantasy"])

    assert result.exit_code == 0
    assert "A hero rises." in result.output
    assert "fantasy" in result.output
    assert "testrun123" in result.output
    assert "genre_normalizer" in result.output
    assert "book_fetcher" in result.output
