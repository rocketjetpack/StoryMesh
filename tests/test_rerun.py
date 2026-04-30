"""Tests for the rerun / regenerate_cover_art functionality."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from storymesh.cli import app
from storymesh.orchestration.pipeline import regenerate_cover_art

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_IMAGE_PROMPT = (
    "Flat 2D cover art, filling the entire canvas. A lone figure at the edge "
    "of a flooded plaza. Gritty ink wash. Palette: charcoal and amber."
)


def _make_proposal_draft_json(**overrides: Any) -> dict[str, Any]:  # noqa: ANN401
    """Build a minimal serialised ProposalDraftAgentOutput for artifact fixtures."""
    proposal: dict[str, Any] = {
        "seed_id": "S1",
        "title": "The Last Inquest",
        "protagonist": (
            "Mara Voss — former census-taker whose faith in due process "
            "survived the collapse even as the process itself did not."
        ),
        "setting": "A flooded city-state where municipal records were lost in year one.",
        "plot_arc": (
            "Act 1: Mara finds a body. Act 2: She rebuilds investigation "
            "infrastructure. Act 3: A community tribunal convicts."
        ),
        "thematic_thesis": "Justice requires witnesses, not institutions.",
        "key_scenes": [
            "Mara finds the body at dawn.",
            "The candlelit tribunal room.",
        ],
        "tensions_addressed": ["T1"],
        "tone": ["dark"],
        "genre_blend": ["mystery", "post_apocalyptic"],
        "image_prompt": _VALID_IMAGE_PROMPT,
    }
    proposal.update(overrides.get("proposal", {}))

    base: dict[str, Any] = {
        "proposal": proposal,
        "all_candidates": [proposal],
        "selection_rationale": {
            "selected_index": 0,
            "rationale": "Only candidate.",
            "runner_up_index": None,
            "cliche_violations": {},
        },
        "debug": {
            "num_candidates_requested": 1,
            "num_valid_candidates": 1,
            "num_parse_failures": 0,
            "total_llm_calls": 1,
            "draft_temperature": 1.2,
            "selection_temperature": 0.2,
        },
        "schema_version": "1.2",
    }
    base.update({k: v for k, v in overrides.items() if k != "proposal"})
    return base


def _write_proposal_draft(run_dir: Path) -> None:
    """Write a valid proposal_draft_output.json into run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    data = json.dumps(_make_proposal_draft_json()).encode()
    (run_dir / "proposal_draft_output.json").write_bytes(data)


# ---------------------------------------------------------------------------
# regenerate_cover_art — error paths
# ---------------------------------------------------------------------------


class TestRegenerateCoverArtErrors:
    def test_no_runs_raises_runtime_error(self, tmp_path: Path) -> None:
        with patch("storymesh.orchestration.pipeline.ArtifactStore") as mock_store_cls:
            mock_store = MagicMock()
            mock_store.list_run_ids.return_value = []
            mock_store_cls.return_value = mock_store

            with pytest.raises(RuntimeError, match="No runs found"):
                regenerate_cover_art()

    def test_missing_proposal_draft_raises_runtime_error(self, tmp_path: Path) -> None:
        with patch("storymesh.orchestration.pipeline.ArtifactStore") as mock_store_cls:
            mock_store = MagicMock()
            mock_store.list_run_ids.return_value = ["abc123"]
            mock_store.load_run_file.return_value = None
            mock_store_cls.return_value = mock_store

            with pytest.raises(RuntimeError, match="proposal_draft_output.json"):
                regenerate_cover_art("abc123")

    def test_no_api_key_raises_value_error(self, tmp_path: Path) -> None:
        import orjson  # noqa: PLC0415

        with (
            patch("storymesh.orchestration.pipeline.ArtifactStore") as mock_store_cls,
            patch("storymesh.orchestration.graph._build_image_client", return_value=None),
            patch("storymesh.config.get_agent_config", return_value={}),
        ):
            mock_store = MagicMock()
            mock_store.list_run_ids.return_value = ["abc123"]
            mock_store.load_run_file.return_value = orjson.dumps(
                _make_proposal_draft_json()
            )
            mock_store_cls.return_value = mock_store

            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                regenerate_cover_art("abc123")

    def test_explicit_run_id_used_when_provided(self, tmp_path: Path) -> None:
        """When a run_id is given, list_run_ids should not be called."""
        with patch("storymesh.orchestration.pipeline.ArtifactStore") as mock_store_cls:
            mock_store = MagicMock()
            mock_store.load_run_file.return_value = None
            mock_store_cls.return_value = mock_store

            with pytest.raises(RuntimeError):
                regenerate_cover_art("specific-run-id")

            mock_store.list_run_ids.assert_not_called()


# ---------------------------------------------------------------------------
# regenerate_cover_art — happy path (mocked image client)
# ---------------------------------------------------------------------------


class TestRegenerateCoverArtHappyPath:
    def test_returns_image_path_string(self, tmp_path: Path) -> None:
        import orjson  # noqa: PLC0415

        fake_client = MagicMock()
        fake_client.model = "gpt-image-2"

        from storymesh.llm.image_base import GeneratedImage  # noqa: PLC0415

        fake_client.generate.return_value = GeneratedImage(
            image_bytes=b"PNG_FAKE", revised_prompt=None
        )

        with (
            patch("storymesh.orchestration.pipeline.ArtifactStore") as mock_store_cls,
            patch(
                "storymesh.orchestration.graph._build_image_client",
                return_value=fake_client,
            ),
            patch(
                "storymesh.config.get_agent_config",
                return_value={"image_provider": "openai", "image_model": "gpt-image-2"},
            ),
            patch("storymesh.core.artifacts.persist_node_output"),
        ):
            mock_store = MagicMock()
            mock_store.list_run_ids.return_value = ["abc123"]
            mock_store.load_run_file.return_value = orjson.dumps(
                _make_proposal_draft_json()
            )
            mock_store.runs_dir = tmp_path / "runs"
            mock_store_cls.return_value = mock_store

            result = regenerate_cover_art("abc123")

        assert isinstance(result, str)
        assert "cover_art.png" in result

    def test_image_saved_via_store(self, tmp_path: Path) -> None:
        import orjson  # noqa: PLC0415

        fake_client = MagicMock()
        fake_client.model = "gpt-image-2"

        from storymesh.llm.image_base import GeneratedImage  # noqa: PLC0415

        fake_client.generate.return_value = GeneratedImage(
            image_bytes=b"PNG_FAKE", revised_prompt=None
        )

        with (
            patch("storymesh.orchestration.pipeline.ArtifactStore") as mock_store_cls,
            patch(
                "storymesh.orchestration.graph._build_image_client",
                return_value=fake_client,
            ),
            patch(
                "storymesh.config.get_agent_config",
                return_value={"image_provider": "openai", "image_model": "gpt-image-2"},
            ),
            patch("storymesh.core.artifacts.persist_node_output"),
        ):
            mock_store = MagicMock()
            mock_store.list_run_ids.return_value = ["abc123"]
            mock_store.load_run_file.return_value = orjson.dumps(
                _make_proposal_draft_json()
            )
            mock_store.runs_dir = tmp_path / "runs"
            mock_store_cls.return_value = mock_store

            regenerate_cover_art("abc123")

        mock_store.save_run_binary.assert_called_once()
        call_args = mock_store.save_run_binary.call_args
        assert call_args[0][0] == "abc123"
        assert call_args[0][1] == "cover_art.png"

    def test_default_run_id_uses_most_recent(self, tmp_path: Path) -> None:
        import orjson  # noqa: PLC0415

        fake_client = MagicMock()
        fake_client.model = "gpt-image-2"

        from storymesh.llm.image_base import GeneratedImage  # noqa: PLC0415

        fake_client.generate.return_value = GeneratedImage(
            image_bytes=b"PNG_FAKE", revised_prompt=None
        )

        with (
            patch("storymesh.orchestration.pipeline.ArtifactStore") as mock_store_cls,
            patch(
                "storymesh.orchestration.graph._build_image_client",
                return_value=fake_client,
            ),
            patch(
                "storymesh.config.get_agent_config",
                return_value={"image_provider": "openai", "image_model": "gpt-image-2"},
            ),
            patch("storymesh.core.artifacts.persist_node_output"),
        ):
            mock_store = MagicMock()
            mock_store.list_run_ids.return_value = ["newest", "older"]
            mock_store.load_run_file.return_value = orjson.dumps(
                _make_proposal_draft_json()
            )
            mock_store.runs_dir = tmp_path / "runs"
            mock_store_cls.return_value = mock_store

            regenerate_cover_art()

        # Should have loaded from the most recent run, not "older"
        mock_store.load_run_file.assert_called_once_with("newest", "proposal_draft_output.json")


# ---------------------------------------------------------------------------
# CLI rerun command
# ---------------------------------------------------------------------------


runner = CliRunner()


class TestRerunCli:
    def test_unsupported_stage_exits_with_error(self) -> None:
        result = runner.invoke(app, ["rerun", "theme_extractor"])
        assert result.exit_code != 0
        assert "theme_extractor" in result.output
        assert "cover_art" in result.output

    def test_cover_art_no_runs_shows_error(self) -> None:
        with patch("storymesh.regenerate_cover_art") as mock_regen:
            mock_regen.side_effect = RuntimeError("No runs found in the artifact store.")
            result = runner.invoke(app, ["rerun", "cover_art"])
        assert result.exit_code != 0
        assert "No runs found" in result.output

    def test_cover_art_no_api_key_shows_error(self) -> None:
        with patch("storymesh.regenerate_cover_art") as mock_regen:
            mock_regen.side_effect = ValueError("OPENAI_API_KEY is not set.")
            result = runner.invoke(app, ["rerun", "cover_art"])
        assert result.exit_code != 0
        assert "OPENAI_API_KEY" in result.output

    def test_cover_art_success_shows_path(self) -> None:
        with patch("storymesh.regenerate_cover_art") as mock_regen:
            mock_regen.return_value = "/home/user/.storymesh/runs/abc123/cover_art.png"
            result = runner.invoke(app, ["rerun", "cover_art"])
        assert result.exit_code == 0
        assert "cover_art.png" in result.output

    def test_cover_art_with_explicit_run_id(self) -> None:
        with patch("storymesh.regenerate_cover_art") as mock_regen:
            mock_regen.return_value = "/home/user/.storymesh/runs/abc123/cover_art.png"
            result = runner.invoke(app, ["rerun", "cover_art", "abc123"])
        assert result.exit_code == 0
        mock_regen.assert_called_once_with("abc123")

    def test_cover_art_without_run_id_passes_none(self) -> None:
        with patch("storymesh.regenerate_cover_art") as mock_regen:
            mock_regen.return_value = "/some/path/cover_art.png"
            runner.invoke(app, ["rerun", "cover_art"])
        mock_regen.assert_called_once_with(None)
