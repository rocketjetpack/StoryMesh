"""Unit tests for storymesh.diagnostics.stylometric_counter.

Each detection rule is tested against known positive cases and innocent
paragraphs to verify the pattern does not over-fire.
"""

from __future__ import annotations

from storymesh.diagnostics.stylometric_counter import (
    count_as_if_as_though,
    count_cascading_which_was,
    count_negation_triplet,
    count_numerical_precision,
    count_proxy_feeling_simile,
    count_tics,
    max_which_was_chain_depth,
    sentence_fragment_paragraph_rate,
)

# ---------------------------------------------------------------------------
# cascading_which_was
# ---------------------------------------------------------------------------


class TestCascadingWhichWas:
    def test_detects_two_clause_sentence(self) -> None:
        text = "She held the form, which was unsigned, which was also undated."
        assert count_cascading_which_was(text) == 1

    def test_detects_three_clause_sentence(self) -> None:
        text = (
            "The building, which was condemned, which had been empty for years, "
            "which was also the only shelter they had, stood at the corner."
        )
        assert count_cascading_which_was(text) == 1

    def test_does_not_count_single_clause(self) -> None:
        text = "The file, which was missing two pages, sat on her desk."
        assert count_cascading_which_was(text) == 0

    def test_counts_multiple_sentences(self) -> None:
        text = (
            "She held the form, which was unsigned, which was also undated. "
            "He read it twice. "
            "The report, which was late, which had no author, vanished the next day."
        )
        assert count_cascading_which_was(text) == 2

    def test_empty_string_returns_zero(self) -> None:
        assert count_cascading_which_was("") == 0

    def test_case_insensitive(self) -> None:
        text = "It was there, WHICH WAS strange, which were all gone."
        assert count_cascading_which_was(text) == 1


# ---------------------------------------------------------------------------
# max_which_was_chain_depth
# ---------------------------------------------------------------------------


class TestMaxWhichWasChainDepth:
    def test_returns_zero_for_no_clauses(self) -> None:
        assert max_which_was_chain_depth("The cat sat on the mat.") == 0

    def test_returns_one_for_single_clause(self) -> None:
        assert max_which_was_chain_depth("The file, which was missing, lay there.") == 1

    def test_returns_three_for_deep_chain(self) -> None:
        text = (
            "The rule, which was unclear, which had been added late, "
            "which was already superseded, caused confusion."
        )
        assert max_which_was_chain_depth(text) == 3

    def test_returns_max_across_sentences(self) -> None:
        text = (
            "She held it, which was warm. "
            "The form, which was blank, which was torn, disappeared."
        )
        assert max_which_was_chain_depth(text) == 2


# ---------------------------------------------------------------------------
# proxy_feeling_simile
# ---------------------------------------------------------------------------


class TestProxyFeelSimile:
    def test_detects_the_way_x_when(self) -> None:
        text = "She knew it the way you know something when you stop asking about it."
        assert count_proxy_feeling_simile(text) == 1

    def test_detects_the_way_x_that(self) -> None:
        text = "He smiled the way strangers do that makes it worse."
        assert count_proxy_feeling_simile(text) == 1

    def test_detects_the_way_x_something(self) -> None:
        text = "It hurt the way something hurts when it's the last time."
        assert count_proxy_feeling_simile(text) == 1

    def test_does_not_match_innocent_the_way(self) -> None:
        text = "She pointed the way down the hall. The way to the exit is on the left."
        # "the way down the hall" — no when/that/something following within range
        assert count_proxy_feeling_simile(text) == 0

    def test_counts_multiple(self) -> None:
        text = (
            "She knew it the way you know something when you stop asking. "
            "He looked the way people look when they've already decided."
        )
        assert count_proxy_feeling_simile(text) == 2

    def test_empty_string_returns_zero(self) -> None:
        assert count_proxy_feeling_simile("") == 0


# ---------------------------------------------------------------------------
# negation_triplet
# ---------------------------------------------------------------------------


class TestNegationTriplet:
    def test_detects_three_not_x_not_y_patterns(self) -> None:
        text = (
            "Not grief, not relief, not the absence of either. "
            "Not a question, not an answer. "
            "Not this, not that."
        )
        assert count_negation_triplet(text) >= 1

    def test_does_not_count_two_negations(self) -> None:
        text = "Not grief, not relief. She carried on."
        assert count_negation_triplet(text) == 0

    def test_empty_string_returns_zero(self) -> None:
        assert count_negation_triplet("") == 0

    def test_counts_per_paragraph(self) -> None:
        para1 = "Not now. Not here. Not like this."
        para2 = "She waited."
        para3 = "Not then either. Not before. Not after."
        text = f"{para1}\n\n{para2}\n\n{para3}"
        assert count_negation_triplet(text) == 2


# ---------------------------------------------------------------------------
# sentence_fragment_paragraph_rate
# ---------------------------------------------------------------------------


class TestSentenceFragmentRate:
    def test_returns_zero_for_long_paragraphs(self) -> None:
        text = "She walked down the hall and opened the heavy oak door. It was dark inside."
        assert sentence_fragment_paragraph_rate(text) == 0.0

    def test_detects_short_paragraphs(self) -> None:
        short = "Three days."
        long = "She had been walking for three days without stopping to eat or sleep."
        text = f"{short}\n\n{long}"
        rate = sentence_fragment_paragraph_rate(text)
        assert rate == 0.5

    def test_all_short_returns_one(self) -> None:
        text = "Three.\n\nFour.\n\nFive."
        assert sentence_fragment_paragraph_rate(text) == 1.0

    def test_empty_string_returns_zero(self) -> None:
        assert sentence_fragment_paragraph_rate("") == 0.0


# ---------------------------------------------------------------------------
# as_if_as_though
# ---------------------------------------------------------------------------


class TestAsIfAsThough:
    def test_counts_as_if(self) -> None:
        assert count_as_if_as_though("She looked as if she hadn't slept.") == 1

    def test_counts_as_though(self) -> None:
        assert count_as_if_as_though("He spoke as though nothing had happened.") == 1

    def test_counts_multiple(self) -> None:
        text = "She acted as if she knew. He responded as though he didn't."
        assert count_as_if_as_though(text) == 2

    def test_case_insensitive(self) -> None:
        assert count_as_if_as_though("AS IF that mattered.") == 1

    def test_does_not_count_partial_match(self) -> None:
        # "assign" contains "as" but not "as if"
        assert count_as_if_as_though("She made an assignment.") == 0

    def test_empty_string_returns_zero(self) -> None:
        assert count_as_if_as_though("") == 0


# ---------------------------------------------------------------------------
# numerical_precision
# ---------------------------------------------------------------------------


class TestNumericalPrecision:
    def test_detects_digit_in_sentence(self) -> None:
        assert count_numerical_precision("She waited 3 days.") >= 1

    def test_detects_spelled_out_integer(self) -> None:
        assert count_numerical_precision("Three days had passed.") >= 1

    def test_does_not_count_question_sentence(self) -> None:
        # Question marks don't end with "."
        assert count_numerical_precision("How many days did she wait?") == 0

    def test_empty_string_returns_zero(self) -> None:
        assert count_numerical_precision("") == 0


# ---------------------------------------------------------------------------
# count_tics — aggregated result
# ---------------------------------------------------------------------------


class TestCountTics:
    def test_returns_expected_keys(self) -> None:
        result = count_tics("She waited. He did not.")
        assert "word_count" in result
        assert "tics" in result
        for key in (
            "cascading_which_was",
            "proxy_feeling_simile",
            "negation_triplet",
            "sentence_fragment_rate",
            "as_if_as_though",
            "numerical_precision",
            "which_was_chain_depth_max",
        ):
            assert key in result["tics"], f"missing tic key: {key}"

    def test_word_count_matches_draft(self) -> None:
        draft = "The body had been arranged with care. Three days ago it was not there."
        result = count_tics(draft)
        assert result["word_count"] == len(draft.split())

    def test_handles_empty_draft(self) -> None:
        result = count_tics("")
        assert result["word_count"] == 0
        assert result["tics"]["cascading_which_was"]["count"] == 0
        assert result["tics"]["as_if_as_though"]["per_1000_words"] == 0.0
        assert result["tics"]["sentence_fragment_rate"]["value"] == 0.0

    def test_handles_unicode_punctuation(self) -> None:
        text = "She held it — the form — which was unsigned, which was also blank."
        result = count_tics(text)
        # No crash; em-dash is handled gracefully
        assert isinstance(result["word_count"], int)

    def test_per_1000_words_is_float(self) -> None:
        draft = " ".join(["word"] * 1000) + ". She looked as if she knew."
        result = count_tics(draft)
        assert isinstance(result["tics"]["as_if_as_though"]["per_1000_words"], float)

    def test_count_based_tics_have_per_1000_words(self) -> None:
        result = count_tics("She looked as if she knew.")
        count_tics_with_rate = {
            k for k, v in result["tics"].items() if "per_1000_words" in v
        }
        assert "as_if_as_though" in count_tics_with_rate

    def test_ratio_max_tics_have_value_key(self) -> None:
        result = count_tics("She walked. The form, which was blank, lay there.")
        assert "value" in result["tics"]["sentence_fragment_rate"]
        assert "value" in result["tics"]["which_was_chain_depth_max"]
