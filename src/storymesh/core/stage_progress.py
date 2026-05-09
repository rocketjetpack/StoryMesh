"""Stage-progress inference shared between the CLI and the kiosk backend.

The pipeline persists a ``{stage}_output.json`` file per stage as each agent
completes (see :mod:`storymesh.core.artifacts`). External observers
(CLI live table, kiosk JSON API) read these files together with the
``llm_calls.jsonl`` tail to infer which stage is currently active.
"""

from __future__ import annotations

from pathlib import Path

import orjson

# Pipeline stage names in execution order. Single source of truth for any
# external code that needs to render or reason about stage progress.
STAGE_NAMES: list[str] = [
    "genre_normalizer",
    "book_fetcher",
    "book_ranker",
    "theme_extractor",
    "proposal_draft",
    "rubric_judge",
    "proposal_reader_feedback",
    "story_writer",
    "resonance_reviewer",  # Stage 6b
    "cover_art",           # Stage 7
    "book_assembler",      # Stage 8
]


def read_last_llm_agent(run_dir: Path) -> str | None:
    """Return the agent name from the last ``llm_calls.jsonl`` line, if any."""
    path = run_dir / "llm_calls.jsonl"
    if not path.exists():
        return None
    lines = path.read_bytes().splitlines()
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            raw = orjson.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(raw, dict):
            return str(raw.get("agent", "")) or None
    return None


def infer_stage_statuses(run_dir: Path | None) -> tuple[dict[str, str], str | None]:
    """Infer per-stage statuses from artifact files and recent LLM activity.

    Returns a ``(statuses, active_stage)`` tuple. ``statuses`` maps every
    name in :data:`STAGE_NAMES` to ``"pending"``, ``"running"``, or ``"done"``.
    ``active_stage`` is the name currently in progress, or ``None`` when the
    run directory has not yet been created.
    """
    statuses = {stage: "pending" for stage in STAGE_NAMES}
    if run_dir is None:
        return statuses, None

    for stage in STAGE_NAMES:
        if (run_dir / f"{stage}_output.json").exists():
            statuses[stage] = "done"

    active_stage = read_last_llm_agent(run_dir)
    if active_stage not in statuses:
        active_stage = None

    if active_stage is None:
        for stage in STAGE_NAMES:
            if statuses[stage] != "done":
                active_stage = stage
                break

    if active_stage is not None and statuses.get(active_stage) != "done":
        statuses[active_stage] = "running"

    return statuses, active_stage
