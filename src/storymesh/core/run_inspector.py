"""Run inspection utilities for StoryMesh.

Provides the ``RunInspector`` class, which loads all available data for a
past pipeline run from disk and exposes it as typed dataclasses.  Also
generates a self-contained HTML report from a loaded ``RunInspection``.

The module is intentionally free of Rich imports so that it can be used
independently of the CLI.  All console rendering lives in ``cli.py``.
"""

from __future__ import annotations

import html
import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import orjson

from storymesh.core.artifacts import ArtifactStore
from storymesh.exceptions import RunNotFoundError

logger = logging.getLogger(__name__)

# Stages in pipeline execution order — mirrors _STAGE_NAMES in cli.py.
_STAGE_NAMES: tuple[str, ...] = (
    "genre_normalizer",
    "book_fetcher",
    "book_ranker",
    "theme_extractor",
    "proposal_draft",
    "rubric_judge",
    "story_writer",
)


class StageStatus(StrEnum):
    """Status of a single pipeline stage artifact on disk."""

    DONE = "done"
    MISSING = "missing"
    CORRUPT = "corrupt"


@dataclass(frozen=True)
class RunMetadata:
    """Metadata written to ``run_metadata.json`` for a pipeline run.

    Attributes:
        user_prompt: The original user-supplied fiction description.
        pipeline_version: StoryMesh version that produced this run.
        timestamp: ISO-8601 run start timestamp (UTC).
        run_id: Unique run identifier (hex UUID).
        stage_timings: Wall-clock seconds per stage, keyed by stage name.
            Empty dict when the run did not complete or used an older version
            that did not persist timings.
    """

    user_prompt: str
    pipeline_version: str
    timestamp: str
    run_id: str
    stage_timings: dict[str, float]


@dataclass(frozen=True)
class StageInspection:
    """Loaded state of one pipeline stage artifact.

    Attributes:
        name: Pipeline stage name (e.g. ``'genre_normalizer'``).
        status: Whether the artifact file was found and parsed successfully.
        data: Parsed JSON dict when ``status`` is ``DONE``; ``None`` otherwise.
        artifact_path: Expected filesystem path of the artifact file.
    """

    name: str
    status: StageStatus
    data: dict[str, Any] | None
    artifact_path: Path


@dataclass(frozen=True)
class LLMCallRecord:
    """A single LLM call record loaded from ``llm_calls.jsonl``.

    Attributes:
        ts: ISO-8601 timestamp of the call.
        agent: Agent name that made the call.
        model: Model identifier used.
        temperature: Sampling temperature, or ``None`` if not recorded.
        attempt: 1-based retry attempt number.
        system_prompt: Full system prompt sent to the model.
        user_prompt: Full user prompt sent to the model.
        raw_response: Complete raw response text from the model.
        parse_success: Whether the response parsed as valid JSON.
        latency_ms: Wall-clock milliseconds for the call, or ``None`` if not recorded.
    """

    ts: str
    agent: str
    model: str
    temperature: float | None
    attempt: int
    system_prompt: str
    user_prompt: str
    raw_response: str
    parse_success: bool
    latency_ms: float | None


@dataclass(frozen=True)
class RunInspection:
    """Complete loaded state for a single pipeline run.

    Attributes:
        run_id: Unique run identifier.
        runs_dir: Parent directory that contains this run's subdirectory.
        metadata: Parsed run metadata, or ``None`` if the file was absent/corrupt.
        stages: Per-stage inspection results, ordered by pipeline execution.
        llm_calls: All LLM call records from ``llm_calls.jsonl`` in order.
    """

    run_id: str
    runs_dir: Path
    metadata: RunMetadata | None
    stages: dict[str, StageInspection]
    llm_calls: list[LLMCallRecord]


class RunInspector:
    """Loads and presents past StoryMesh pipeline run data.

    Args:
        store: The ``ArtifactStore`` instance pointing to the data directory.
    """

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, run_id: str) -> RunInspection:
        """Load all available data for a run from disk.

        Never raises for missing or corrupt individual stage files — those are
        represented as ``StageStatus.MISSING`` or ``StageStatus.CORRUPT``
        within the returned ``RunInspection``.

        Args:
            run_id: Run identifier, or ``"latest"`` to use the most recently
                modified run.

        Returns:
            A fully populated ``RunInspection`` dataclass.

        Raises:
            RunNotFoundError: If ``run_id`` does not correspond to an existing
                run directory (or ``"latest"`` is requested but no runs exist).
        """
        resolved = self._resolve_run_id(run_id)
        run_dir = self._store.runs_dir / resolved

        metadata = self._load_metadata(run_dir)
        stages = {name: self._load_stage(run_dir, name) for name in _STAGE_NAMES}
        llm_calls = self._load_llm_calls(run_dir)

        return RunInspection(
            run_id=resolved,
            runs_dir=self._store.runs_dir,
            metadata=metadata,
            stages=stages,
            llm_calls=llm_calls,
        )

    def generate_html(self, report: RunInspection) -> str:
        """Generate a self-contained HTML report for a run inspection.

        The output is a single HTML file with embedded CSS and no external
        dependencies.  ``<details>``/``<summary>`` elements provide
        collapsible sections without JavaScript.  All user-supplied content
        is escaped with ``html.escape`` to prevent injection.

        Args:
            report: A ``RunInspection`` as returned by :meth:`load`.

        Returns:
            Complete HTML document as a string (UTF-8 compatible).
        """
        meta = report.metadata
        title = html.escape(f"StoryMesh Run {report.run_id}")
        prompt_escaped = html.escape(meta.user_prompt if meta else "")
        version = html.escape(meta.pipeline_version if meta else "unknown")
        timestamp = html.escape(meta.timestamp if meta else "unknown")

        css = _HTML_CSS
        stage_sections = "\n".join(
            _html_stage_section(s) for s in report.stages.values()
        )
        llm_sections = "\n".join(_html_llm_section(c) for c in report.llm_calls)
        llm_block = (
            llm_sections
            if llm_sections
            else "<p><em>No LLM calls recorded for this run.</em></p>"
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>{css}</style>
</head>
<body>
  <header>
    <h1>StoryMesh Run Report</h1>
    <dl>
      <dt>Run ID</dt><dd><code>{html.escape(report.run_id)}</code></dd>
      <dt>Prompt</dt><dd>{prompt_escaped}</dd>
      <dt>Version</dt><dd>{version}</dd>
      <dt>Timestamp</dt><dd>{timestamp}</dd>
    </dl>
  </header>

  <section id="stages">
    <h2>Pipeline Stages</h2>
    {stage_sections}
  </section>

  <section id="llm-calls">
    <h2>LLM Calls</h2>
    {llm_block}
  </section>
</body>
</html>"""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_run_id(self, run_id: str) -> str:
        """Resolve ``'latest'`` to the most-recent run ID, or validate the given ID.

        Args:
            run_id: Literal run ID string or ``"latest"``.

        Returns:
            Resolved run ID string.

        Raises:
            RunNotFoundError: If the run directory does not exist.
        """
        if run_id == "latest":
            ids = self._store.list_run_ids()
            if not ids:
                raise RunNotFoundError(
                    "No runs found in the runs directory. "
                    "Run 'storymesh generate <prompt>' first."
                )
            return ids[0]

        run_dir = self._store.runs_dir / run_id
        if not run_dir.is_dir():
            raise RunNotFoundError(
                f"Run '{run_id}' not found in {self._store.runs_dir}."
            )
        return run_id

    def _load_metadata(self, run_dir: Path) -> RunMetadata | None:
        """Load ``run_metadata.json``, returning ``None`` on any failure.

        Args:
            run_dir: Absolute path to the run directory.

        Returns:
            Parsed ``RunMetadata``, or ``None`` if the file is absent or corrupt.
        """
        path = run_dir / "run_metadata.json"
        if not path.exists():
            return None
        try:
            raw: Any = orjson.loads(path.read_bytes())
            if not isinstance(raw, dict):
                logger.warning("run_metadata.json is not a dict in %s", run_dir)
                return None
            timings_raw = raw.get("stage_timings", {})
            timings: dict[str, float] = (
                timings_raw if isinstance(timings_raw, dict) else {}
            )
            return RunMetadata(
                user_prompt=str(raw.get("user_prompt", "")),
                pipeline_version=str(raw.get("pipeline_version", "unknown")),
                timestamp=str(raw.get("timestamp", "")),
                run_id=str(raw.get("run_id", "")),
                stage_timings=timings,
            )
        except Exception:
            logger.warning("Could not parse run_metadata.json in %s", run_dir, exc_info=True)
            return None

    def _load_stage(self, run_dir: Path, stage_name: str) -> StageInspection:
        """Load one stage artifact file.

        Args:
            run_dir: Absolute path to the run directory.
            stage_name: Pipeline stage name (e.g. ``'genre_normalizer'``).

        Returns:
            ``StageInspection`` with ``DONE``, ``MISSING``, or ``CORRUPT`` status.
        """
        path = run_dir / f"{stage_name}_output.json"
        if not path.exists():
            return StageInspection(
                name=stage_name,
                status=StageStatus.MISSING,
                data=None,
                artifact_path=path,
            )
        try:
            raw: Any = orjson.loads(path.read_bytes())
            if not isinstance(raw, dict):
                logger.warning(
                    "Stage artifact %s is not a JSON object in %s", stage_name, run_dir
                )
                return StageInspection(
                    name=stage_name,
                    status=StageStatus.CORRUPT,
                    data=None,
                    artifact_path=path,
                )
            return StageInspection(
                name=stage_name,
                status=StageStatus.DONE,
                data=raw,
                artifact_path=path,
            )
        except Exception:
            logger.warning(
                "Could not parse stage artifact %s in %s", stage_name, run_dir, exc_info=True
            )
            return StageInspection(
                name=stage_name,
                status=StageStatus.CORRUPT,
                data=None,
                artifact_path=path,
            )

    def _load_llm_calls(self, run_dir: Path) -> list[LLMCallRecord]:
        """Load all LLM call records from ``llm_calls.jsonl``.

        Each line is parsed independently; corrupt lines are skipped with a
        warning so that a single bad line does not prevent the rest from loading.

        Args:
            run_dir: Absolute path to the run directory.

        Returns:
            List of ``LLMCallRecord`` instances in file order.
        """
        path = run_dir / "llm_calls.jsonl"
        if not path.exists():
            return []

        records: list[LLMCallRecord] = []
        for line_num, line in enumerate(path.read_bytes().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw: Any = orjson.loads(line)
                if not isinstance(raw, dict):
                    logger.warning("llm_calls.jsonl line %d is not a dict", line_num)
                    continue
                records.append(
                    LLMCallRecord(
                        ts=str(raw.get("ts", "")),
                        agent=str(raw.get("agent", "unknown")),
                        model=str(raw.get("model", "unknown")),
                        temperature=_as_float(raw.get("temperature")),
                        attempt=int(raw.get("attempt", 1)),
                        system_prompt=str(raw.get("system_prompt", "")),
                        user_prompt=str(raw.get("user_prompt", "")),
                        raw_response=str(raw.get("raw_response", "")),
                        parse_success=bool(raw.get("parse_success", False)),
                        latency_ms=_as_float(raw.get("latency_ms")),
                    )
                )
            except Exception:
                logger.warning(
                    "Could not parse llm_calls.jsonl line %d in %s",
                    line_num,
                    run_dir,
                    exc_info=True,
                )
        return records


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _as_float(value: object) -> float | None:
    """Convert a value to float, returning None if conversion fails."""
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _html_stage_section(stage: StageInspection) -> str:
    """Build an HTML ``<details>`` block for one pipeline stage."""
    status_label = {
        StageStatus.DONE: "&#10003; done",
        StageStatus.MISSING: "&#9675; missing",
        StageStatus.CORRUPT: "&#10007; corrupt",
    }[stage.status]

    status_class = {
        StageStatus.DONE: "status-done",
        StageStatus.MISSING: "status-missing",
        StageStatus.CORRUPT: "status-corrupt",
    }[stage.status]

    summary = (
        f'<summary><span class="stage-name">{html.escape(stage.name)}</span> '
        f'<span class="{status_class}">{status_label}</span></summary>'
    )

    if stage.status == StageStatus.MISSING:
        body = "<p><em>Stage not yet run.</em></p>"
    elif stage.status == StageStatus.CORRUPT:
        body = "<p class='status-corrupt'><em>Artifact file could not be parsed.</em></p>"
    else:
        body = _html_stage_body(stage.name, stage.data or {})

    open_attr = " open" if stage.status == StageStatus.DONE else ""
    return f"<details{open_attr}>{summary}\n{body}\n</details>"


def _html_stage_body(name: str, data: dict[str, Any]) -> str:
    """Render stage-specific content as HTML."""
    if name == "genre_normalizer":
        return _html_genre_normalizer(data)
    if name == "book_fetcher":
        return _html_book_fetcher(data)
    if name == "book_ranker":
        return _html_book_ranker(data)
    if name == "theme_extractor":
        return _html_theme_extractor(data)
    if name == "proposal_draft":
        return _html_proposal_draft(data)
    # Generic fallback for future stages
    escaped = html.escape(str(data)[:2000])
    return f"<pre>{escaped}</pre>"


def _html_genre_normalizer(data: dict[str, Any]) -> str:
    genres = ", ".join(str(g) for g in _list(data.get("normalized_genres"))) or "—"
    subgenres = ", ".join(str(g) for g in _list(data.get("subgenres"))) or "—"
    tones = ", ".join(str(t) for t in _list(data.get("user_tones"))) or "—"
    inferred_raw = _list(data.get("inferred_genres"))
    inferred = ", ".join(
        str(g.get("canonical_genre", g)) if isinstance(g, dict) else str(g)
        for g in inferred_raw
    ) or "—"
    context = ", ".join(str(c) for c in _list(data.get("narrative_context"))) or "—"
    override_note = data.get("override_note") or ""
    override_html = (
        f"<p class='override-note'>{html.escape(str(override_note))}</p>"
        if override_note
        else ""
    )
    return (
        f"<dl>"
        f"<dt>Genres</dt><dd>{html.escape(genres)}</dd>"
        f"<dt>Subgenres</dt><dd>{html.escape(subgenres)}</dd>"
        f"<dt>Tones</dt><dd>{html.escape(tones)}</dd>"
        f"<dt>Inferred genres</dt><dd>{html.escape(inferred)}</dd>"
        f"<dt>Narrative context</dt><dd>{html.escape(context)}</dd>"
        f"</dl>{override_html}"
    )


def _html_book_fetcher(data: dict[str, Any]) -> str:
    queries = ", ".join(str(q) for q in _list(data.get("queries_executed"))) or "—"
    books = _list(data.get("books"))
    rows = ""
    for book in books[:5]:
        if not isinstance(book, dict):
            continue
        title = html.escape(str(book.get("title", "—")))
        authors = html.escape(", ".join(str(a) for a in _list(book.get("authors"))))
        year = html.escape(str(book.get("first_publish_year", "—")))
        rows += f"<tr><td>{title}</td><td>{authors}</td><td>{year}</td></tr>"
    table = (
        f"<table><thead><tr><th>Title</th><th>Authors</th><th>Year</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        if rows
        else "<p><em>No books recorded.</em></p>"
    )
    return (
        f"<dl><dt>Queries</dt><dd>{html.escape(queries)}</dd>"
        f"<dt>Total books fetched</dt><dd>{len(books)}</dd></dl>"
        f"<h4>Top books (up to 5)</h4>{table}"
    )


def _html_book_ranker(data: dict[str, Any]) -> str:
    ranked = _list(data.get("ranked_books"))
    dropped = data.get("dropped_count", 0)
    rows = ""
    for rb in ranked[:5]:
        if not isinstance(rb, dict):
            continue
        book = rb.get("book", {})
        if not isinstance(book, dict):
            continue
        rank = html.escape(str(rb.get("rank", "—")))
        title = html.escape(str(book.get("title", "—")))
        authors = html.escape(", ".join(str(a) for a in _list(book.get("authors"))))
        score = html.escape(f"{float(rb.get('composite_score', 0)):.3f}")
        breakdown = rb.get("score_breakdown", {})
        if isinstance(breakdown, dict):
            go = html.escape(f"{float(breakdown.get('genre_overlap', 0)):.3f}")
            re = html.escape(f"{float(breakdown.get('reader_engagement', 0)):.3f}")
        else:
            go = re = "—"
        rows += (
            f"<tr><td>{rank}</td><td>{title}</td><td>{authors}</td>"
            f"<td>{score}</td><td>{go}</td><td>{re}</td></tr>"
        )
    table = (
        f"<table><thead><tr>"
        f"<th>Rank</th><th>Title</th><th>Authors</th>"
        f"<th>Score</th><th>Genre Overlap</th><th>Reader Engagement</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
        if rows
        else "<p><em>No ranked books recorded.</em></p>"
    )
    return (
        f"<dl><dt>Books below cutoff</dt><dd>{html.escape(str(dropped))}</dd></dl>"
        f"<h4>Top ranked books (up to 5)</h4>{table}"
    )


def _html_theme_extractor(data: dict[str, Any]) -> str:
    clusters = _list(data.get("genre_clusters"))
    cluster_items = ""
    for c in clusters:
        if not isinstance(c, dict):
            continue
        genre = html.escape(str(c.get("genre", "—")))
        n_books = len(_list(c.get("books")))
        assumptions = "; ".join(str(a) for a in _list(c.get("thematic_assumptions")))
        cluster_items += (
            f"<li><strong>{genre}</strong> ({n_books} books) — "
            f"{html.escape(assumptions)}</li>"
        )
    cluster_html = f"<ul>{cluster_items}</ul>" if cluster_items else "<p>—</p>"

    tensions = _list(data.get("tensions"))
    tension_items = ""
    for t in tensions:
        if not isinstance(t, dict):
            continue
        tid = html.escape(str(t.get("tension_id", "—")))
        cq = html.escape(str(t.get("creative_question", "")))
        intensity = t.get("intensity", 0)
        tension_items += (
            f"<li><strong>{tid}</strong> (intensity {float(intensity):.2f}) — {cq}</li>"
        )
    tension_html = f"<ul>{tension_items}</ul>" if tension_items else "<p>—</p>"

    seeds = _list(data.get("narrative_seeds"))
    seed_items = ""
    for s in seeds:
        if not isinstance(s, dict):
            continue
        sid = html.escape(str(s.get("seed_id", "—")))
        concept = html.escape(str(s.get("concept", "")))
        seed_items += f"<li><strong>{sid}</strong> — {concept}</li>"
    seed_html = f"<ul>{seed_items}</ul>" if seed_items else "<p>—</p>"

    return (
        f"<h4>Genre Clusters</h4>{cluster_html}"
        f"<h4>Thematic Tensions</h4>{tension_html}"
        f"<h4>Narrative Seeds</h4>{seed_html}"
    )


def _html_proposal_draft(data: dict[str, Any]) -> str:
    proposal = data.get("proposal", {})
    if not isinstance(proposal, dict):
        proposal = {}
    rationale = data.get("selection_rationale", {})
    if not isinstance(rationale, dict):
        rationale = {}
    debug = data.get("debug", {})
    if not isinstance(debug, dict):
        debug = {}

    title = html.escape(str(proposal.get("title", "—")))
    protagonist = html.escape(str(proposal.get("protagonist", "—")))
    setting = html.escape(str(proposal.get("setting", "—")))
    thematic_thesis = html.escape(str(proposal.get("thematic_thesis", "—")))
    tone = html.escape(str(proposal.get("tone", "—")))
    genre_blend = html.escape(
        ", ".join(str(g) for g in _list(proposal.get("genre_blend"))) or "—"
    )
    plot_arc = html.escape(str(proposal.get("plot_arc", "—")))
    key_scenes = _list(proposal.get("key_scenes"))
    scenes_items = "".join(f"<li>{html.escape(str(s))}</li>" for s in key_scenes)
    scenes_html = f"<ol>{scenes_items}</ol>" if scenes_items else "<p>—</p>"

    sel_index = rationale.get("selected_index", "—")
    n_valid = debug.get("num_valid_candidates", "—")
    runner_up = rationale.get("runner_up_index")
    runner_up_str = f", runner-up: {runner_up}" if runner_up is not None else ""
    rationale_text = html.escape(str(rationale.get("rationale", "—")))
    cliche_violations = _list(rationale.get("cliche_violations"))
    cliche_html = ""
    if cliche_violations:
        items = "".join(f"<li>{html.escape(str(v))}</li>" for v in cliche_violations)
        cliche_html = f"<h4>Cliché Violations</h4><ul>{items}</ul>"

    n_requested = debug.get("num_candidates_requested", "—")
    n_failures = debug.get("num_parse_failures", 0)
    draft_temp = debug.get("draft_temperature", "—")
    sel_temp = debug.get("selection_temperature", "—")
    total_calls = debug.get("total_llm_calls", "—")

    return (
        f"<h4>Selected Proposal</h4>"
        f"<dl>"
        f"<dt>Title</dt><dd><strong>{title}</strong></dd>"
        f"<dt>Protagonist</dt><dd>{protagonist}</dd>"
        f"<dt>Setting</dt><dd>{setting}</dd>"
        f"<dt>Tone</dt><dd>{tone}</dd>"
        f"<dt>Genre blend</dt><dd>{genre_blend}</dd>"
        f"<dt>Thematic thesis</dt><dd>{thematic_thesis}</dd>"
        f"<dt>Plot arc</dt><dd>{plot_arc}</dd>"
        f"</dl>"
        f"<h4>Key Scenes</h4>{scenes_html}"
        f"<h4>Selection Rationale</h4>"
        f"<dl>"
        f"<dt>Selected</dt><dd>candidate {sel_index} of {n_valid}{runner_up_str}</dd>"
        f"<dt>Rationale</dt><dd>{rationale_text}</dd>"
        f"</dl>"
        f"{cliche_html}"
        f"<h4>Debug</h4>"
        f"<dl>"
        f"<dt>Candidates requested / valid / failed</dt>"
        f"<dd>{n_requested} / {n_valid} / {n_failures}</dd>"
        f"<dt>Draft temperature</dt><dd>{draft_temp}</dd>"
        f"<dt>Selection temperature</dt><dd>{sel_temp}</dd>"
        f"<dt>Total LLM calls</dt><dd>{total_calls}</dd>"
        f"</dl>"
    )


def _html_llm_section(call: LLMCallRecord) -> str:
    """Build a collapsible HTML block for one LLM call record."""
    latency = f"{call.latency_ms:.0f}ms" if call.latency_ms is not None else "?"
    parse_icon = "&#10003;" if call.parse_success else "&#10007;"
    parse_class = "status-done" if call.parse_success else "status-corrupt"
    summary = (
        f"<summary>"
        f"[{html.escape(call.agent)}] attempt {call.attempt} &mdash; "
        f"{latency} &mdash; "
        f'<span class="{parse_class}">{parse_icon} parse</span>'
        f"</summary>"
    )
    sys_pre = f"<pre>{html.escape(call.system_prompt)}</pre>"
    usr_pre = f"<pre>{html.escape(call.user_prompt)}</pre>"
    rsp_pre = f"<pre>{html.escape(call.raw_response)}</pre>"
    return (
        f"<details>"
        f"{summary}"
        f"<h4>System Prompt</h4>{sys_pre}"
        f"<h4>User Prompt</h4>{usr_pre}"
        f"<h4>Response</h4>{rsp_pre}"
        f"</details>"
    )


def _list(value: object) -> list[Any]:
    """Return ``value`` as a list, or an empty list if it is not a list."""
    return value if isinstance(value, list) else []


_HTML_CSS = """
body {
  font-family: system-ui, sans-serif;
  max-width: 960px;
  margin: 2rem auto;
  padding: 0 1rem;
  color: #1a1a1a;
  line-height: 1.5;
}
header { border-bottom: 2px solid #ccc; margin-bottom: 1.5rem; padding-bottom: 1rem; }
h1 { margin: 0 0 0.5rem; font-size: 1.5rem; }
h2 { font-size: 1.2rem; margin-top: 2rem; }
h4 { margin: 0.75rem 0 0.25rem; font-size: 0.95rem; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 0.2rem 1rem; }
dt { font-weight: 600; color: #555; }
details { border: 1px solid #ddd; border-radius: 4px; margin: 0.5rem 0; padding: 0.5rem; }
details[open] { background: #fafafa; }
summary { cursor: pointer; font-weight: 600; padding: 0.2rem 0; list-style: none; }
summary::-webkit-details-marker { display: none; }
.stage-name { font-family: monospace; }
.status-done  { color: #2a7a2a; }
.status-missing { color: #888; }
.status-corrupt { color: #c0392b; }
.override-note { color: #b7600a; font-style: italic; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; margin: 0.5rem 0; }
th, td { border: 1px solid #ddd; padding: 0.3rem 0.5rem; text-align: left; }
th { background: #f0f0f0; }
pre {
  background: #f5f5f5;
  border: 1px solid #ddd;
  border-radius: 3px;
  padding: 0.75rem;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 0.82rem;
  max-height: 400px;
  overflow-y: auto;
}
ul { margin: 0.25rem 0; padding-left: 1.5rem; }
li { margin: 0.2rem 0; }
code { font-family: monospace; background: #f0f0f0; padding: 0.1em 0.3em; border-radius: 3px; }
"""
