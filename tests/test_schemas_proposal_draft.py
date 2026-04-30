"""Unit tests for storymesh.schemas.proposal_draft."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from storymesh.schemas.proposal_draft import (
    ProposalDraftAgentInput,
    ProposalDraftAgentOutput,
    SelectionRationale,
    StoryProposal,
)
from storymesh.schemas.theme_extractor import (
    GenreCluster,
    NarrativeSeed,
    ThematicTension,
)
from storymesh.versioning.schemas import PROPOSAL_SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LONG_PLOT_ARC = (
    "Act 1: A former homicide detective, now working as a scavenger in a "
    "collapsed city-state, discovers a body arranged with deliberate symbolism. "
    "Act 2: Her investigation forces her to rebuild the very infrastructure of "
    "truth-gathering — witnesses, records, trust — from scratch, while a rival "
    "faction uses the crime to consolidate power. Act 3: She exposes the killer "
    "but cannot prosecute; instead she engineers a public reckoning that costs "
    "her the only community she has rebuilt."
)


def _cluster(**overrides: object) -> GenreCluster:
    defaults: dict[str, object] = {
        "genre": "mystery",
        "books": ["The Big Sleep"],
        "thematic_assumptions": ["Truth is recoverable through investigation"],
    }
    return GenreCluster(**{**defaults, **overrides})


def _tension(**overrides: object) -> ThematicTension:
    defaults: dict[str, object] = {
        "tension_id": "T1",
        "cluster_a": "mystery",
        "assumption_a": "Truth is recoverable",
        "cluster_b": "post_apocalyptic",
        "assumption_b": "Records and institutions no longer exist",
        "creative_question": "What does investigation mean without infrastructure?",
        "intensity": 0.9,
        "cliched_resolutions": [
            "A lone detective rebuilds justice through sheer determination",
        ],
    }
    return ThematicTension(**{**defaults, **overrides})


def _seed(**overrides: object) -> NarrativeSeed:
    defaults: dict[str, object] = {
        "seed_id": "S1",
        "concept": "A scavenger detective reinvents investigation in a collapsed city.",
        "tensions_used": ["T1"],
    }
    return NarrativeSeed(**{**defaults, **overrides})


def _proposal(**overrides: object) -> dict:
    defaults: dict[str, object] = {
        "seed_id": "S1",
        "title": "The Last Inquest",
        "protagonist": "Mara Voss — a former homicide detective whose faith in due process "
                       "survived the collapse even as the process itself did not.",
        "setting": "A flooded mid-21st-century city-state where municipal records were lost "
                   "in the first year of collapse.",
        "plot_arc": _LONG_PLOT_ARC,
        "thematic_thesis": (
            "Justice does not require institutions to be meaningful, "
            "but meaning without institutions cannot produce justice."
        ),
        "key_scenes": [
            "Mara finds the arranged body and recognises the killer's signature.",
            "She convenes a community tribunal with no legal authority.",
            "The tribunal reaches a verdict she must choose whether to enforce.",
        ],
        "tensions_addressed": ["T1"],
        "tone": ["dark", "cerebral"],
        "genre_blend": ["mystery", "post_apocalyptic"],
        "image_prompt": (
            "A rain-slicked street in a flooded cityscape at dusk, a lone figure "
            "silhouetted against the pale ruins of a collapsed civic tower. "
            "Gritty noir ink wash style, muted greys and a single amber light source."
        ),
    }
    return {**defaults, **overrides}


def _rationale(**overrides: object) -> dict:
    defaults: dict[str, object] = {
        "selected_index": 0,
        "rationale": "Candidate 0 avoids all flagged clichés and has the sharpest thematic thesis.",
        "cliche_violations": {},
        "runner_up_index": None,
    }
    return {**defaults, **overrides}


def _valid_output() -> ProposalDraftAgentOutput:
    proposal = StoryProposal(**_proposal())
    return ProposalDraftAgentOutput(
        proposal=proposal,
        all_candidates=[proposal],
        selection_rationale=SelectionRationale(**_rationale()),
    )


# ---------------------------------------------------------------------------
# TestStoryProposal
# ---------------------------------------------------------------------------


class TestStoryProposal:
    def test_valid_construction(self) -> None:
        p = StoryProposal(**_proposal())
        assert p.title == "The Last Inquest"
        assert p.seed_id == "S1"

    def test_frozen(self) -> None:
        p = StoryProposal(**_proposal())
        with pytest.raises(ValidationError):
            p.title = "Other Title"  # type: ignore[misc]

    def test_title_min_length(self) -> None:
        with pytest.raises(ValidationError):
            StoryProposal(**_proposal(title=""))

    def test_protagonist_min_length(self) -> None:
        with pytest.raises(ValidationError):
            StoryProposal(**_proposal(protagonist="Too short"))

    def test_setting_min_length(self) -> None:
        with pytest.raises(ValidationError):
            StoryProposal(**_proposal(setting="Too short"))

    def test_plot_arc_min_length(self) -> None:
        with pytest.raises(ValidationError):
            StoryProposal(**_proposal(plot_arc="Short."))

    def test_thematic_thesis_min_length(self) -> None:
        with pytest.raises(ValidationError):
            StoryProposal(**_proposal(thematic_thesis="Too short"))

    def test_key_scenes_min_length(self) -> None:
        with pytest.raises(ValidationError):
            StoryProposal(**_proposal(key_scenes=["Only one scene."]))

    def test_tensions_addressed_min_length(self) -> None:
        with pytest.raises(ValidationError):
            StoryProposal(**_proposal(tensions_addressed=[]))

    def test_tone_min_length(self) -> None:
        with pytest.raises(ValidationError):
            StoryProposal(**_proposal(tone=[]))

    def test_genre_blend_min_length(self) -> None:
        with pytest.raises(ValidationError):
            StoryProposal(**_proposal(genre_blend=[]))

    def test_image_prompt_required(self) -> None:
        data = _proposal()
        del data["image_prompt"]
        with pytest.raises(ValidationError):
            StoryProposal(**data)

    def test_image_prompt_min_length(self) -> None:
        with pytest.raises(ValidationError):
            StoryProposal(**_proposal(image_prompt="Too short prompt."))


# ---------------------------------------------------------------------------
# TestSelectionRationale
# ---------------------------------------------------------------------------


class TestSelectionRationale:
    def test_valid_construction(self) -> None:
        r = SelectionRationale(**_rationale())
        assert r.selected_index == 0
        assert r.runner_up_index is None

    def test_frozen(self) -> None:
        r = SelectionRationale(**_rationale())
        with pytest.raises(ValidationError):
            r.selected_index = 1  # type: ignore[misc]

    def test_selected_index_ge_zero(self) -> None:
        with pytest.raises(ValidationError):
            SelectionRationale(**_rationale(selected_index=-1))

    def test_rationale_min_length(self) -> None:
        with pytest.raises(ValidationError):
            SelectionRationale(**_rationale(rationale="Too short"))

    def test_runner_up_defaults_to_none(self) -> None:
        r = SelectionRationale(
            selected_index=0,
            rationale="Best thematic depth and fewest clichés of the three candidates.",
        )
        assert r.runner_up_index is None

    def test_runner_up_can_be_set(self) -> None:
        r = SelectionRationale(**_rationale(runner_up_index=1))
        assert r.runner_up_index == 1

    def test_cliche_violations_defaults_to_empty(self) -> None:
        r = SelectionRationale(
            selected_index=0,
            rationale="Best thematic depth and fewest clichés of the three candidates.",
        )
        assert r.cliche_violations == {}

    def test_cliche_violations_accepts_mapping(self) -> None:
        r = SelectionRationale(
            **_rationale(cliche_violations={"0": [], "1": ["Hero saves the day"]})
        )
        assert r.cliche_violations["1"] == ["Hero saves the day"]


# ---------------------------------------------------------------------------
# TestProposalDraftAgentInput
# ---------------------------------------------------------------------------


class TestProposalDraftAgentInput:
    def test_valid_construction(self) -> None:
        inp = ProposalDraftAgentInput(
            narrative_seeds=[_seed()],
            tensions=[_tension()],
            genre_clusters=[_cluster()],
            normalized_genres=["mystery"],
            user_prompt="dark post-apocalyptic mystery",
        )
        assert inp.user_prompt == "dark post-apocalyptic mystery"

    def test_empty_seeds_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProposalDraftAgentInput(
                narrative_seeds=[],
                tensions=[_tension()],
                genre_clusters=[_cluster()],
                normalized_genres=["mystery"],
                user_prompt="test",
            )

    def test_empty_tensions_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProposalDraftAgentInput(
                narrative_seeds=[_seed()],
                tensions=[],
                genre_clusters=[_cluster()],
                normalized_genres=["mystery"],
                user_prompt="test",
            )

    def test_empty_clusters_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProposalDraftAgentInput(
                narrative_seeds=[_seed()],
                tensions=[_tension()],
                genre_clusters=[],
                normalized_genres=["mystery"],
                user_prompt="test",
            )

    def test_empty_genres_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProposalDraftAgentInput(
                narrative_seeds=[_seed()],
                tensions=[_tension()],
                genre_clusters=[_cluster()],
                normalized_genres=[],
                user_prompt="test",
            )

    def test_defaults_for_optional_fields(self) -> None:
        inp = ProposalDraftAgentInput(
            narrative_seeds=[_seed()],
            tensions=[_tension()],
            genre_clusters=[_cluster()],
            normalized_genres=["mystery"],
            user_prompt="test",
        )
        assert inp.user_tones == []
        assert inp.narrative_context == []


# ---------------------------------------------------------------------------
# TestProposalDraftAgentOutput
# ---------------------------------------------------------------------------


class TestProposalDraftAgentOutput:
    def test_valid_construction(self) -> None:
        output = _valid_output()
        assert isinstance(output.proposal, StoryProposal)
        assert len(output.all_candidates) == 1

    def test_frozen(self) -> None:
        output = _valid_output()
        with pytest.raises(ValidationError):
            output.proposal = StoryProposal(**_proposal(title="Other"))  # type: ignore[misc]

    def test_schema_version_matches(self) -> None:
        output = _valid_output()
        assert output.schema_version == PROPOSAL_SCHEMA_VERSION

    def test_all_candidates_min_length(self) -> None:
        proposal = StoryProposal(**_proposal())
        with pytest.raises(ValidationError):
            ProposalDraftAgentOutput(
                proposal=proposal,
                all_candidates=[],
                selection_rationale=SelectionRationale(**_rationale()),
            )

    def test_debug_defaults_to_empty_dict(self) -> None:
        output = _valid_output()
        assert output.debug == {}

    def test_debug_accepts_metadata(self) -> None:
        proposal = StoryProposal(**_proposal())
        output = ProposalDraftAgentOutput(
            proposal=proposal,
            all_candidates=[proposal],
            selection_rationale=SelectionRationale(**_rationale()),
            debug={"num_candidates_requested": 3, "total_llm_calls": 4},
        )
        assert output.debug["num_candidates_requested"] == 3
