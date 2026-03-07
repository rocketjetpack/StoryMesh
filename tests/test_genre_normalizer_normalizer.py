"""
Unit tests for the storymesh.agents.genre_normalizer.normalize module.
"""

from storymesh.agents.genre_normalizer.normalize import normalize_text


class TestNormalizeText:
    def test_lowercase(self) -> None:
        assert normalize_text("UPPERCASE") == "uppercase"

    def test_hyphen_replacement(self) -> None:
        assert normalize_text("sci-fi") == "sci fi"
    
    def test_multiple_hyphen_replacement(self) -> None:
        assert normalize_text("sci-fi-action") == "sci fi action"

    def test_underscore_replacement(self) -> None:
        assert normalize_text("action_adventure") == "action adventure"

    def test_multiple_whitespace(self) -> None:
        assert normalize_text("  multiple   spaces  ") == "multiple spaces"

    def test_strip_leading_trailing_whitespace(self) -> None:
        assert normalize_text("  leading and trailing  ") == "leading and trailing"

    def test_combined_transformations(self) -> None:
        assert normalize_text(" Sci_fi-Action  adventure") == "sci fi action adventure"

    def text_comma_removal(self) -> None:
        assert normalize_text("sci-fi, action") == "sci fi action"

    def test_no_normalization_needed(self) -> None:
        assert normalize_text("normal text") == "normal text"

    def test_strip_quotes(self) -> None:
        assert normalize_text('"quoted text"') == "quoted text"

    def test_replace_ampersand(self) -> None:
        assert normalize_text("sci fi & mystery") == "sci fi and mystery"

    def test_replace_parentheses(self) -> None:
        assert normalize_text("action (adventure)") == "action adventure"