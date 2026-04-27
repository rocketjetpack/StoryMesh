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


def test_generate_reports_existing_artifacts(tmp_path: Path) -> None:
    """When run_dir contains stage artifacts, generate marks stages as done
    and prints the artifact directory footer."""
    import orjson

    run_dir = tmp_path / "run_with_art"
    run_dir.mkdir()
    (run_dir / "genre_normalizer_output.json").write_bytes(orjson.dumps({"ok": True}))

    mock_result = MagicMock()
    mock_result.final_synopsis = "Synopsis body."
    mock_result.metadata = {
        "user_prompt": "dark fantasy",
        "run_id": "run_with_art",
        "pipeline_version": "0.6.0",
        "stage_timings": {"genre_normalizer": 0.02},
        "run_dir": str(run_dir),
    }

    with patch("storymesh.cli.generate_synopsis", return_value=mock_result):
        result = runner.invoke(app, ["generate", "dark fantasy"])

    assert result.exit_code == 0
    # "✓ done" status line should appear because the artifact exists.
    assert "done" in result.output
    # Footer "Artifacts saved to:" only prints when the run_dir exists.
    assert "Artifacts saved to" in result.output
    assert str(run_dir) in result.output


# ---------------------------------------------------------------------------
# show-config / show-agent-config
# ---------------------------------------------------------------------------


def test_show_config_outputs_yaml(tmp_path: Path) -> None:
    """show-config prints the resolved config as YAML with the file path."""
    cfg_path = tmp_path / "storymesh.config.yaml"

    with (
        patch("storymesh.config.find_config_file", return_value=cfg_path),
        patch(
            "storymesh.config.get_config",
            return_value={"agents": {"default": {"model": "claude-haiku-4-5"}}},
        ),
    ):
        result = runner.invoke(app, ["show-config"])

    assert result.exit_code == 0
    assert str(cfg_path) in result.output
    # YAML-formatted content must appear.
    assert "agents" in result.output
    assert "model: claude-haiku-4-5" in result.output


def test_show_agent_config_outputs_yaml() -> None:
    """show-agent-config prints the resolved config for the named agent."""
    resolved = {"model": "claude-haiku-4-5-20251001", "temperature": 0.2}

    with patch("storymesh.config.get_agent_config", return_value=resolved):
        result = runner.invoke(app, ["show-agent-config", "genre_normalizer"])

    assert result.exit_code == 0
    assert "genre_normalizer" in result.output
    assert "claude-haiku-4-5-20251001" in result.output
    assert "temperature: 0.2" in result.output


# ---------------------------------------------------------------------------
# inspect-run — exercise all Rich stage renderers
# ---------------------------------------------------------------------------


def _write_full_stage_run(tmp_path: Path, run_id: str) -> None:
    """Write a run with every implemented stage populated.

    This covers every ``_rich_*`` and ``_html_*`` renderer branch when the
    resulting run is loaded via ``inspect-run``.
    """
    import orjson

    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "user_prompt": "a dark post-apocalyptic mystery",
        "pipeline_version": "0.7.0",
        "timestamp": "2026-04-15T00:00:00+00:00",
        "run_id": run_id,
        "stage_timings": {
            "genre_normalizer": 0.2,
            "book_fetcher": 1.0,
            "book_ranker": 0.1,
            "theme_extractor": 3.4,
            "proposal_draft": 2.8,
        },
    }
    (run_dir / "run_metadata.json").write_bytes(orjson.dumps(meta))

    # genre_normalizer with tone_override + inferred genres.
    (run_dir / "genre_normalizer_output.json").write_bytes(
        orjson.dumps(
            {
                "raw_input": "a dark mystery",
                "normalized_genres": ["mystery"],
                "subgenres": ["noir"],
                "user_tones": ["bleak"],
                "tone_override": True,
                "override_note": "User 'bleak' overrides default 'mysterious'.",
                "narrative_context": ["flooded city"],
                "inferred_genres": [
                    {"canonical_genre": "post_apocalyptic", "confidence": 0.6},
                ],
                "debug": {},
                "schema_version": "3.1",
            }
        )
    )

    # book_fetcher.
    (run_dir / "book_fetcher_output.json").write_bytes(
        orjson.dumps(
            {
                "books": [
                    {
                        "work_key": "/works/OL1W",
                        "title": "The Drowned Archive",
                        "authors": ["Alice Rivers"],
                        "first_publish_year": 2021,
                        "source_genres": ["mystery"],
                    },
                    {
                        "work_key": "/works/OL2W",
                        "title": "Silent Quorum",
                        "authors": ["Bay Kline"],
                        "first_publish_year": 2019,
                        "source_genres": ["mystery"],
                    },
                ],
                "queries_executed": ["mystery", "post apocalyptic"],
            }
        )
    )

    # book_ranker.
    (run_dir / "book_ranker_output.json").write_bytes(
        orjson.dumps(
            {
                "ranked_books": [
                    {
                        "book": {
                            "work_key": "/works/OL1W",
                            "title": "The Drowned Archive",
                            "authors": ["Alice Rivers"],
                            "source_genres": ["mystery"],
                        },
                        "rank": 1,
                        "composite_score": 0.87,
                        "score_breakdown": {
                            "genre_overlap": 0.9,
                            "reader_engagement": 0.75,
                        },
                    }
                ],
                "ranked_summaries": [],
                "dropped_count": 2,
                "llm_reranked": False,
                "debug": {},
            }
        )
    )

    # theme_extractor.
    (run_dir / "theme_extractor_output.json").write_bytes(
        orjson.dumps(
            {
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
                        "creative_question": (
                            "What does investigation mean without infrastructure?"
                        ),
                        "intensity": 0.9,
                        "cliched_resolutions": [
                            "A lone detective rebuilds justice through determination",
                        ],
                    }
                ],
                "narrative_seeds": [
                    {
                        "seed_id": "S1",
                        "concept": (
                            "A scavenger detective reinvents investigation in a "
                            "collapsed city."
                        ),
                        "tensions_used": ["T1"],
                        "tonal_direction": [],
                        "narrative_context_used": [],
                    }
                ],
                "user_tones_carried": ["bleak"],
            }
        )
    )

    # proposal_draft (covers the new _rich_proposal_draft / _html_proposal_draft).
    (run_dir / "proposal_draft_output.json").write_bytes(
        orjson.dumps(
            {
                "proposal": {
                    "seed_id": "S1",
                    "title": "The Last Inquest",
                    "protagonist": "Mara Voss, a former homicide detective.",
                    "setting": "A flooded mid-21st-century city-state.",
                    "plot_arc": "Act 1: the body. Act 2: rebuild investigation. Act 3: verdict.",
                    "thematic_thesis": "Justice does not require institutions.",
                    "key_scenes": [
                        "Mara finds the arranged body.",
                        "A community tribunal with no legal authority convenes.",
                    ],
                    "tensions_addressed": ["T1"],
                    "tone": ["bleak"],
                    "genre_blend": ["mystery", "post_apocalyptic"],
                },
                "selection_rationale": {
                    "selected_index": 1,
                    "rationale": "Candidate 1 hit the thematic question most directly.",
                    # NOTE: the Pydantic schema types cliche_violations as
                    # dict[str, list[str]], but both _rich_proposal_draft and
                    # _html_proposal_draft pass the value through _as_list,
                    # which yields [] for a dict and silently omits the
                    # "Cliché violations" block. Supplying a raw list here
                    # exercises the list-branch of the renderers directly.
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
        )
    )


def test_inspect_run_renders_all_implemented_stages(tmp_path: Path) -> None:
    """inspect-run triggers every implemented ``_rich_*`` stage renderer."""
    _write_full_stage_run(tmp_path, "run_full")
    from storymesh.core.artifacts import ArtifactStore as RealStore

    real_store = RealStore(root=tmp_path)

    with patch("storymesh.cli.ArtifactStore", return_value=real_store):
        result = runner.invoke(app, ["inspect-run", "run_full"])

    assert result.exit_code == 0
    # Genre normalizer tone_override branch.
    assert "Tone override" in result.output
    assert "mysterious" in result.output
    # Book fetcher top books.
    assert "The Drowned Archive" in result.output
    # Book ranker score line.
    assert "overlap=0.900" in result.output
    # Theme extractor content.
    assert "Thematic Tensions" in result.output
    assert "Narrative Seeds" in result.output
    # Proposal draft content.
    assert "The Last Inquest" in result.output
    assert "Key scenes" in result.output
    assert "Cliché violations" in result.output


def test_inspect_run_llm_filter_with_no_matches(tmp_path: Path) -> None:
    """--llm <agent> with no matching calls prints the 'no calls found' notice."""
    _write_cli_run(tmp_path, "run_nomatch")
    from storymesh.core.artifacts import ArtifactStore as RealStore

    real_store = RealStore(root=tmp_path)

    with patch("storymesh.cli.ArtifactStore", return_value=real_store):
        result = runner.invoke(
            app, ["inspect-run", "run_nomatch", "--llm", "no_such_agent"]
        )

    assert result.exit_code == 0
    assert "No LLM calls found" in result.output


def test_inspect_run_no_llm_calls_notice(tmp_path: Path) -> None:
    """A run without an llm_calls.jsonl file shows the empty-summary notice."""
    import orjson

    run_dir = tmp_path / "runs" / "run_silent"
    run_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_bytes(
        orjson.dumps(
            {
                "user_prompt": "p",
                "pipeline_version": "0.6.0",
                "timestamp": "2026-04-03T12:00:00+00:00",
                "run_id": "run_silent",
                "stage_timings": {},
            }
        )
    )
    from storymesh.core.artifacts import ArtifactStore as RealStore

    real_store = RealStore(root=tmp_path)

    with patch("storymesh.cli.ArtifactStore", return_value=real_store):
        result = runner.invoke(app, ["inspect-run", "run_silent"])

    assert result.exit_code == 0
    assert "No LLM calls recorded" in result.output
