"""Unit tests for storymesh.prompts.loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from storymesh.prompts.loader import (
    PromptFormattingError,
    PromptTemplate,
    get_prompt_style,
    load_prompt,
    set_prompt_style,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_prompt_yaml(
    path: Path,
    system: str = "You are a test assistant.",
    user: str = "Classify: {tokens}",
) -> Path:
    """Write a minimal prompt YAML file and return the path."""
    data = {"system": system, "user": user}
    path.write_text(yaml.dump(data, default_flow_style=False))
    return path


# ---------------------------------------------------------------------------
# PromptTemplate — construction
# ---------------------------------------------------------------------------

class TestPromptTemplateConstruction:
    def test_valid_construction(self) -> None:
        pt = PromptTemplate(system="system text", user_template="user {placeholder}")
        assert pt.system == "system text"

    def test_empty_system_raises(self) -> None:
        with pytest.raises(ValueError, match="System prompt"):
            PromptTemplate(system="", user_template="user text")

    def test_whitespace_system_raises(self) -> None:
        with pytest.raises(ValueError, match="System prompt"):
            PromptTemplate(system="   ", user_template="user text")

    def test_empty_user_template_raises(self) -> None:
        with pytest.raises(ValueError, match="User prompt template"):
            PromptTemplate(system="system text", user_template="")

    def test_whitespace_user_template_raises(self) -> None:
        with pytest.raises(ValueError, match="User prompt template"):
            PromptTemplate(system="system text", user_template="   ")


# ---------------------------------------------------------------------------
# PromptTemplate — format_user()
# ---------------------------------------------------------------------------

class TestFormatUser:
    def test_basic_formatting(self) -> None:
        pt = PromptTemplate(system="sys", user_template="Hello {name}, you have {count} items.")
        result = pt.format_user(name="Alice", count=3)
        assert result == "Hello Alice, you have 3 items."

    def test_missing_placeholder_raises(self) -> None:
        pt = PromptTemplate(system="sys", user_template="Hello {name}, task: {task}")
        with pytest.raises(PromptFormattingError, match="name"):
            pt.format_user(task="classify")

    def test_error_message_includes_provided_keys(self) -> None:
        pt = PromptTemplate(system="sys", user_template="{a} {b} {c}")
        with pytest.raises(PromptFormattingError, match="Provided") as exc_info:
            pt.format_user(a="1", c="3")
        assert "b" in str(exc_info.value)

    def test_extra_kwargs_ignored(self) -> None:
        pt = PromptTemplate(system="sys", user_template="Hello {name}")
        result = pt.format_user(name="Alice", extra="ignored")
        assert result == "Hello Alice"

    def test_format_with_list_values(self) -> None:
        pt = PromptTemplate(
            system="sys",
            user_template="Genres: {resolved_genres}\nText: {remaining_text}",
        )
        result = pt.format_user(
            resolved_genres=["fantasy", "mystery"],
            remaining_text="about a rebellion",
        )
        assert "['fantasy', 'mystery']" in result
        assert "about a rebellion" in result

    def test_system_prompt_unchanged(self) -> None:
        """System prompt with curly braces should not be affected by format_user."""
        system = 'Return JSON: {"key": "value"}'
        pt = PromptTemplate(system=system, user_template="Hello {name}")
        pt.format_user(name="Alice")
        assert pt.system == system


# ---------------------------------------------------------------------------
# load_prompt() — valid files
# ---------------------------------------------------------------------------

class TestLoadPrompt:
    def test_loads_valid_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        default_dir = tmp_path / "styles" / "default"
        default_dir.mkdir(parents=True)
        _write_prompt_yaml(default_dir / "test_agent.yaml")
        monkeypatch.setattr("storymesh.prompts.loader._PROMPTS_DIR", tmp_path)
        monkeypatch.setattr("storymesh.prompts.loader._STYLES_DIR", tmp_path / "styles")

        pt = load_prompt("test_agent")
        assert pt.system == "You are a test assistant."

    def test_format_user_after_load(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        default_dir = tmp_path / "styles" / "default"
        default_dir.mkdir(parents=True)
        _write_prompt_yaml(default_dir / "test_agent.yaml", user="Classify: {tokens}")
        monkeypatch.setattr("storymesh.prompts.loader._PROMPTS_DIR", tmp_path)
        monkeypatch.setattr("storymesh.prompts.loader._STYLES_DIR", tmp_path / "styles")

        pt = load_prompt("test_agent")
        result = pt.format_user(tokens="rebellion 2085")
        assert "rebellion 2085" in result

    def test_loads_style_specific_prompt(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        styles_dir = tmp_path / "styles" / "slim"
        styles_dir.mkdir(parents=True)
        _write_prompt_yaml(styles_dir / "test_agent.yaml", system="Slim system")
        monkeypatch.setattr("storymesh.prompts.loader._PROMPTS_DIR", tmp_path)
        monkeypatch.setattr("storymesh.prompts.loader._STYLES_DIR", tmp_path / "styles")

        pt = load_prompt("test_agent", style="slim")
        assert pt.system == "Slim system"

    def test_style_falls_back_to_default_style_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        default_dir = tmp_path / "styles" / "default"
        default_dir.mkdir(parents=True)
        _write_prompt_yaml(default_dir / "test_agent.yaml", system="Default style system")
        monkeypatch.setattr("storymesh.prompts.loader._PROMPTS_DIR", tmp_path)
        monkeypatch.setattr("storymesh.prompts.loader._STYLES_DIR", tmp_path / "styles")

        pt = load_prompt("test_agent", style="slim")
        assert pt.system == "Default style system"

    def test_style_missing_and_default_missing_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("storymesh.prompts.loader._PROMPTS_DIR", tmp_path)
        monkeypatch.setattr("storymesh.prompts.loader._STYLES_DIR", tmp_path / "styles")

        with pytest.raises(FileNotFoundError, match="test_agent"):
            load_prompt("test_agent", style="slim")

    def test_active_prompt_style_is_used(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        styles_dir = tmp_path / "styles" / "slim"
        styles_dir.mkdir(parents=True)
        _write_prompt_yaml(styles_dir / "test_agent.yaml", system="Active slim system")
        monkeypatch.setattr("storymesh.prompts.loader._PROMPTS_DIR", tmp_path)
        monkeypatch.setattr("storymesh.prompts.loader._STYLES_DIR", tmp_path / "styles")

        original_style = get_prompt_style()
        try:
            set_prompt_style("slim")
            pt = load_prompt("test_agent")
        finally:
            set_prompt_style(original_style)

        assert pt.system == "Active slim system"


# ---------------------------------------------------------------------------
# load_prompt() — error cases
# ---------------------------------------------------------------------------

class TestLoadPromptErrors:
    def test_missing_file_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("storymesh.prompts.loader._PROMPTS_DIR", tmp_path)
        monkeypatch.setattr("storymesh.prompts.loader._STYLES_DIR", tmp_path / "styles")
        with pytest.raises(FileNotFoundError, match="nonexistent_agent"):
            load_prompt("nonexistent_agent")

    def test_missing_system_key_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        default_dir = tmp_path / "styles" / "default"
        default_dir.mkdir(parents=True)
        path = default_dir / "bad_agent.yaml"
        path.write_text(yaml.dump({"user": "some template"}))
        monkeypatch.setattr("storymesh.prompts.loader._PROMPTS_DIR", tmp_path)
        monkeypatch.setattr("storymesh.prompts.loader._STYLES_DIR", tmp_path / "styles")

        with pytest.raises(ValueError, match="system"):
            load_prompt("bad_agent")

    def test_missing_user_key_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        default_dir = tmp_path / "styles" / "default"
        default_dir.mkdir(parents=True)
        path = default_dir / "bad_agent.yaml"
        path.write_text(yaml.dump({"system": "some system prompt"}))
        monkeypatch.setattr("storymesh.prompts.loader._PROMPTS_DIR", tmp_path)
        monkeypatch.setattr("storymesh.prompts.loader._STYLES_DIR", tmp_path / "styles")

        with pytest.raises(ValueError, match="user"):
            load_prompt("bad_agent")

    def test_empty_system_value_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        default_dir = tmp_path / "styles" / "default"
        default_dir.mkdir(parents=True)
        path = default_dir / "bad_agent.yaml"
        path.write_text(yaml.dump({"system": "", "user": "template"}))
        monkeypatch.setattr("storymesh.prompts.loader._PROMPTS_DIR", tmp_path)
        monkeypatch.setattr("storymesh.prompts.loader._STYLES_DIR", tmp_path / "styles")

        with pytest.raises(ValueError, match="system"):
            load_prompt("bad_agent")

    def test_empty_user_value_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        default_dir = tmp_path / "styles" / "default"
        default_dir.mkdir(parents=True)
        path = default_dir / "bad_agent.yaml"
        path.write_text(yaml.dump({"system": "valid", "user": ""}))
        monkeypatch.setattr("storymesh.prompts.loader._PROMPTS_DIR", tmp_path)
        monkeypatch.setattr("storymesh.prompts.loader._STYLES_DIR", tmp_path / "styles")

        with pytest.raises(ValueError, match="user"):
            load_prompt("bad_agent")

    def test_non_dict_yaml_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        default_dir = tmp_path / "styles" / "default"
        default_dir.mkdir(parents=True)
        path = default_dir / "bad_agent.yaml"
        path.write_text("- just\n- a\n- list\n")
        monkeypatch.setattr("storymesh.prompts.loader._PROMPTS_DIR", tmp_path)
        monkeypatch.setattr("storymesh.prompts.loader._STYLES_DIR", tmp_path / "styles")

        with pytest.raises(ValueError, match="mapping"):
            load_prompt("bad_agent")

    def test_invalid_yaml_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        default_dir = tmp_path / "styles" / "default"
        default_dir.mkdir(parents=True)
        path = default_dir / "bad_agent.yaml"
        path.write_text("system: [\ninvalid yaml")
        monkeypatch.setattr("storymesh.prompts.loader._PROMPTS_DIR", tmp_path)
        monkeypatch.setattr("storymesh.prompts.loader._STYLES_DIR", tmp_path / "styles")

        with pytest.raises(ValueError, match="Invalid YAML"):
            load_prompt("bad_agent")

    def test_invalid_style_name_raises(self) -> None:
        with pytest.raises(ValueError, match="path separators"):
            load_prompt("genre_normalizer", style="../slim")


# ---------------------------------------------------------------------------
# Integration: load the real genre_normalizer.yaml
# ---------------------------------------------------------------------------

class TestLoadGenreNormalizerPrompt:
    def test_loads_successfully(self) -> None:
        """Verify the actual genre_normalizer.yaml file loads without error."""
        pt = load_prompt("genre_normalizer")
        assert len(pt.system) > 0

    def test_user_template_has_expected_placeholders(self) -> None:
        """Verify the user template accepts the kwargs that resolve_llm will provide."""
        pt = load_prompt("genre_normalizer")
        result = pt.format_user(
            raw_input="gritty science fiction about a rebellion in 2085",
            resolved_genres=["science_fiction"],
            resolved_tones=["gritty"],
            remaining_text="about a rebellion in 2085",
        )
        assert "science_fiction" in result
        assert "gritty" in result
        assert "about a rebellion in 2085" in result


# ---------------------------------------------------------------------------
# Integration: load the real proposal_draft_generate.yaml
# ---------------------------------------------------------------------------


class TestProposalDraftGeneratePrompt:
    def test_load_prompt_succeeds(self) -> None:
        """Verify proposal_draft_generate.yaml loads without error."""
        pt = load_prompt("proposal_draft_generate")
        assert pt is not None

    def test_system_prompt_non_empty(self) -> None:
        pt = load_prompt("proposal_draft_generate")
        assert pt.system.strip() != ""

    def test_user_template_has_required_placeholders(self) -> None:
        pt = load_prompt("proposal_draft_generate")
        required = {
            "candidate_index",
            "total_candidates",
            "alternate_angle_note",
            "user_prompt",
            "normalized_genres",
            "user_tones",
            "narrative_context",
            "assigned_seed",
            "additional_seeds",
            "tensions",
            "genre_clusters",
        }
        for placeholder in required:
            assert f"{{{placeholder}}}" in pt._user_template, (
                f"Missing placeholder: {{{placeholder}}}"
            )

    def test_schema_contains_image_prompt(self) -> None:
        pt = load_prompt("proposal_draft_generate")
        assert "image_prompt" in pt.system

    def test_system_contains_step_4(self) -> None:
        pt = load_prompt("proposal_draft_generate")
        assert "STEP 4" in pt.system

    def test_format_user_with_valid_data(self) -> None:
        pt = load_prompt("proposal_draft_generate")
        result = pt.format_user(
            candidate_index=1,
            total_candidates=3,
            alternate_angle_note="",
            user_prompt="dark post-apocalyptic mystery",
            normalized_genres=["mystery", "post_apocalyptic"],
            user_tones=["dark"],
            narrative_context=["flooded city"],
            assigned_seed='{"seed_id": "S1", "concept": "A detective..."}',
            additional_seeds="[]",
            tensions="[]",
            genre_clusters="[]",
        )
        assert "dark post-apocalyptic mystery" in result
        assert "candidate 1 of 3" in result.lower()


# ---------------------------------------------------------------------------
# Integration: load the real proposal_draft_select.yaml
# ---------------------------------------------------------------------------


class TestProposalDraftSelectPrompt:
    def test_load_prompt_succeeds(self) -> None:
        """Verify proposal_draft_select.yaml loads without error."""
        pt = load_prompt("proposal_draft_select")
        assert pt is not None

    def test_system_prompt_non_empty(self) -> None:
        pt = load_prompt("proposal_draft_select")
        assert pt.system.strip() != ""

    def test_user_template_has_required_placeholders(self) -> None:
        pt = load_prompt("proposal_draft_select")
        required = {"user_prompt", "user_tones", "tensions", "candidates"}
        for placeholder in required:
            assert f"{{{placeholder}}}" in pt._user_template, (
                f"Missing placeholder: {{{placeholder}}}"
            )

    def test_format_user_with_valid_data(self) -> None:
        pt = load_prompt("proposal_draft_select")
        result = pt.format_user(
            user_prompt="dark post-apocalyptic mystery",
            user_tones=["dark"],
            tensions="[]",
            candidates="[]",
        )
        assert "dark post-apocalyptic mystery" in result


# ---------------------------------------------------------------------------
# Integration: load the real proposal_draft_retry.yaml
# ---------------------------------------------------------------------------


class TestProposalDraftRetryPrompt:
    def test_load_prompt_succeeds(self) -> None:
        """Verify proposal_draft_retry.yaml loads without error."""
        pt = load_prompt("proposal_draft_retry")
        assert pt is not None

    def test_system_prompt_non_empty(self) -> None:
        pt = load_prompt("proposal_draft_retry")
        assert pt.system.strip() != ""

    def test_system_prompt_matches_generate(self) -> None:
        """Retry system prompt must be identical to generate system prompt."""
        gen = load_prompt("proposal_draft_generate")
        ret = load_prompt("proposal_draft_retry")
        assert ret.system == gen.system

    def test_user_template_has_standard_placeholders(self) -> None:
        """All original generate-prompt placeholders must be present."""
        pt = load_prompt("proposal_draft_retry")
        standard = {
            "candidate_index",
            "total_candidates",
            "alternate_angle_note",
            "user_prompt",
            "normalized_genres",
            "user_tones",
            "narrative_context",
            "assigned_seed",
            "additional_seeds",
            "tensions",
            "genre_clusters",
        }
        for placeholder in standard:
            assert f"{{{placeholder}}}" in pt._user_template, (
                f"Missing standard placeholder: {{{placeholder}}}"
            )

    def test_user_template_has_retry_placeholders(self) -> None:
        """Retry-specific placeholders must be present."""
        pt = load_prompt("proposal_draft_retry")
        retry_specific = {"previous_proposal", "rubric_feedback", "rubric_scores", "attempt_number"}
        for placeholder in retry_specific:
            assert f"{{{placeholder}}}" in pt._user_template, (
                f"Missing retry placeholder: {{{placeholder}}}"
            )

    def test_schema_contains_image_prompt(self) -> None:
        pt = load_prompt("proposal_draft_retry")
        assert "image_prompt" in pt.system

    def test_system_contains_step_4(self) -> None:
        pt = load_prompt("proposal_draft_retry")
        assert "STEP 4" in pt.system

    def test_format_user_with_valid_retry_data(self) -> None:
        pt = load_prompt("proposal_draft_retry")
        result = pt.format_user(
            attempt_number=2,
            previous_proposal='{"title": "Old Title"}',
            rubric_feedback="[tension_inhabitation] (score: 0.3): Tension was resolved.",
            rubric_scores="  tension_inhabitation: 0.3\n  COMPOSITE: 0.45",
            candidate_index=1,
            total_candidates=3,
            alternate_angle_note="",
            user_prompt="dark post-apocalyptic mystery",
            normalized_genres=["mystery", "post_apocalyptic"],
            user_tones=["dark"],
            narrative_context=["flooded city"],
            assigned_seed='{"seed_id": "S1", "concept": "A detective..."}',
            additional_seeds="[]",
            tensions="[]",
            genre_clusters="[]",
        )
        assert "attempt 2" in result.lower()
        assert "dark post-apocalyptic mystery" in result
        assert "Old Title" in result


# ---------------------------------------------------------------------------
# Integration: load the real rubric_judge.yaml
# ---------------------------------------------------------------------------


class TestRubricJudgePrompt:
    def test_load_prompt_succeeds(self) -> None:
        """Verify rubric_judge.yaml loads without error."""
        pt = load_prompt("rubric_judge")
        assert pt is not None

    def test_system_prompt_non_empty(self) -> None:
        pt = load_prompt("rubric_judge")
        assert pt.system.strip() != ""

    def test_system_prompt_contains_all_dimension_names(self) -> None:
        pt = load_prompt("rubric_judge")
        expected = [
            "restraint",
            "story_serving_choices",
            "specificity",
            "protagonist_interiority",
            "user_intent_fidelity",
        ]
        for name in expected:
            assert name in pt.system, f"Dimension '{name}' missing from system prompt"

    def test_system_prompt_contains_principle_refs(self) -> None:
        pt = load_prompt("rubric_judge")
        for ref in [
            "restraint",
            "story_serving_choices",
            "specificity",
            "protagonist_interiority",
            "user_intent_fidelity",
        ]:
            assert ref in pt.system, f"Principle ref '{ref}' not referenced in rubric system prompt"

    def test_user_template_has_required_placeholders(self) -> None:
        pt = load_prompt("rubric_judge")
        required = {"user_prompt", "normalized_genres", "user_tones", "tensions", "proposal"}
        for placeholder in required:
            assert f"{{{placeholder}}}" in pt._user_template, (
                f"Missing placeholder: {{{placeholder}}}"
            )

    def test_format_user_with_valid_data(self) -> None:
        pt = load_prompt("rubric_judge")
        result = pt.format_user(
            user_prompt="dark post-apocalyptic mystery",
            normalized_genres=["mystery", "post_apocalyptic"],
            user_tones=["dark"],
            tensions="[]",
            proposal='{"title": "The Last Inquest"}',
        )
        assert "dark post-apocalyptic mystery" in result
        assert "The Last Inquest" in result


class TestSlimPromptStyle:
    def test_slim_style_loads_style_specific_prompt(self) -> None:
        pt = load_prompt("story_writer_draft", style="slim")
        assert "serve the story rather than the prompt" in (
            pt.system + "\n" + pt._user_template
        ).lower()

    def test_slim_style_falls_back_for_missing_prompt(self) -> None:
        pt = load_prompt("story_writer_summary", style="slim")
        assert "back-cover copy" in pt.system.lower()
