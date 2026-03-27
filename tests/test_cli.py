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
