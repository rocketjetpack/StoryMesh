"""Shared LLM usage aggregation helpers.

Reads ``llm_calls.jsonl`` records written by ``ArtifactStore.log_llm_call``
and produces aggregate token / latency / call-count summaries.

Used by both the CLI run summary and the BookAssembler run-info page so
that figures match exactly across surfaces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import orjson

if TYPE_CHECKING:
    from storymesh.core.artifacts import ArtifactStore


def load_llm_usage_summary(
    store: ArtifactStore,
    run_id: str,
    *,
    include_agents: set[str] | None = None,
    exclude_agents: set[str] | None = None,
) -> dict[str, int]:
    """Summarize rough LLM usage from ``llm_calls.jsonl`` for one run.

    Args:
        store: Artifact store providing access to the run directory.
        run_id: Run identifier whose ``llm_calls.jsonl`` will be aggregated.
        include_agents: When provided, only records whose ``agent`` field is
            in this set are counted. ``None`` (default) means count all.
        exclude_agents: When provided, records whose ``agent`` field is in
            this set are skipped. Applied after ``include_agents``.

    Returns:
        A dict with keys ``calls``, ``approx_prompt_tokens``,
        ``approx_response_tokens``, ``approx_total_tokens``,
        ``parse_failures``, ``latency_ms``. All zero when the file is
        absent or empty.
    """
    raw = store.load_run_file(run_id, "llm_calls.jsonl")
    if raw is None:
        return {
            "calls": 0,
            "approx_prompt_tokens": 0,
            "approx_response_tokens": 0,
            "approx_total_tokens": 0,
            "parse_failures": 0,
            "latency_ms": 0,
        }

    calls = 0
    prompt_tokens = 0
    response_tokens = 0
    total_tokens = 0
    parse_failures = 0
    latency_ms = 0

    for line in raw.splitlines():
        if not line.strip():
            continue
        record = orjson.loads(line)
        if not isinstance(record, dict):
            continue
        agent = str(record.get("agent", ""))
        if include_agents is not None and agent not in include_agents:
            continue
        if exclude_agents is not None and agent in exclude_agents:
            continue
        calls += 1
        prompt_tokens += int(record.get("approx_prompt_tokens", 0) or 0)
        response_tokens += int(record.get("approx_response_tokens", 0) or 0)
        total_tokens += int(record.get("approx_total_tokens", 0) or 0)
        latency_ms += int(record.get("latency_ms", 0) or 0)
        if not bool(record.get("parse_success", False)):
            parse_failures += 1

    return {
        "calls": calls,
        "approx_prompt_tokens": prompt_tokens,
        "approx_response_tokens": response_tokens,
        "approx_total_tokens": total_tokens,
        "parse_failures": parse_failures,
        "latency_ms": latency_ms,
    }
