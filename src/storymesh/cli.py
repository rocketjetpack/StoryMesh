import random
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import orjson
import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from storymesh import generate_synopsis
from storymesh.config import get_config
from storymesh.core.artifacts import ArtifactStore
from storymesh.core.run_inspector import (
    LLMCallRecord,
    RunInspection,
    RunInspector,
    StageInspection,
    StageStatus,
)
from storymesh.exceptions import RunNotFoundError
from storymesh.llm.base import current_run_id
from storymesh.versioning import AGENT_VERSIONS, SCHEMA_VERSIONS
from storymesh.versioning import __version__ as storymesh_version

app = typer.Typer()
console = Console()

# Quality presets: (pass_threshold, max_retries, min_retries, enable_resonance_review)
_QUALITY_PRESETS: dict[str, tuple[int, int, int, bool]] = {
    "draft":     (5, 1, 0, False),
    "standard":  (6, 2, 1, False),
    "high":      (7, 3, 1, True),   # threshold lowered 8→7: rubric scores cluster at 6-7
    "very_high": (9, 3, 2, True),
}

# Pipeline stage names in execution order — used to build the stage table.
_STAGE_NAMES = [
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

_BASELINE_SYSTEM_PROMPT = """\
You are a fiction writer. Write one complete short story in a single pass.

Priorities:
1. Satisfy the user's request faithfully.
2. Tell a coherent, specific story with natural prose.
3. Aim for approximately the requested target length.
4. Prefer concrete action, image, dialogue, and scene over abstract explanation.
5. Let the story carry its themes implicitly rather than explaining them directly.

Return ONLY a valid JSON object with this shape:
{
  "full_draft": "<the complete short story>"
}
"""

_BASELINE_USER_TEMPLATE = """\
USER PROMPT: "{user_prompt}"

TARGET LENGTH: approximately {target_words} words

Write a complete short story that fulfills the user's request. The prose should
feel natural and human-written. Return only the JSON object.
"""


def _build_blinded_eval_packet(
    *,
    run_id: str,
    user_prompt: str,
    storymesh_draft: str,
    storymesh_word_count: int,
    baseline_draft: str,
    baseline_word_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a randomized A/B packet plus a separate answer key."""
    candidates = [
        {
            "id": "A",
            "source": "storymesh",
            "full_draft": storymesh_draft,
            "word_count": storymesh_word_count,
        },
        {
            "id": "B",
            "source": "baseline",
            "full_draft": baseline_draft,
            "word_count": baseline_word_count,
        },
    ]
    random.SystemRandom().shuffle(candidates)

    blinded_candidates = [
        {
            "id": candidate["id"],
            "full_draft": candidate["full_draft"],
            "word_count": candidate["word_count"],
        }
        for candidate in candidates
    ]

    packet = {
        "packet_version": "1.0",
        "run_id": run_id,
        "user_prompt": user_prompt,
        "instructions": (
            "Two candidate stories were generated from the same user prompt. "
            "Evaluate them without assuming which system produced which story."
        ),
        "candidates": blinded_candidates,
    }
    answer_key = {
        "packet_version": "1.0",
        "run_id": run_id,
        "mapping": {
            candidate["id"]: candidate["source"] for candidate in candidates
        },
    }
    return packet, answer_key


def _format_duration(seconds: float) -> str:
    """Render seconds as a compact human-friendly duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {int(rem):02d}s"
    hours, rem = divmod(minutes, 60)
    return f"{int(hours)}h {int(rem):02d}m"


def _run_with_live_status[T](label: str, fn: Callable[[], T]) -> tuple[T, float]:
    """Run a callable in a worker thread while showing elapsed time in the CLI."""
    result_box: dict[str, Any] = {}
    error_box: dict[str, BaseException] = {}

    def _target() -> None:
        try:
            result_box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001
            error_box["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    t0 = time.perf_counter()
    thread.start()

    with console.status(f"{label} [dim]{_format_duration(0.0)}[/dim]") as status:
        while thread.is_alive():
            elapsed = time.perf_counter() - t0
            status.update(f"{label} [dim]{_format_duration(elapsed)}[/dim]")
            thread.join(timeout=0.1)

    elapsed = time.perf_counter() - t0
    if "error" in error_box:
        raise error_box["error"]
    return result_box["value"], elapsed


def _find_active_run_dir(store: ArtifactStore, started_at: float) -> Path | None:
    """Best-effort detection of the run directory being created right now."""
    for run_id in store.list_run_ids():
        candidate = store.runs_dir / run_id
        try:
            if candidate.stat().st_mtime >= started_at - 1.0:
                return candidate
        except FileNotFoundError:
            continue
    return None


def _read_last_llm_agent(run_dir: Path) -> str | None:
    """Return the agent name from the last llm_calls.jsonl line, if any."""
    path = run_dir / "llm_calls.jsonl"
    if not path.exists():
        return None
    lines = path.read_bytes().splitlines()
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            raw = orjson.loads(line)
            if isinstance(raw, dict):
                return str(raw.get("agent", "")) or None
        except Exception:
            continue
    return None


def _infer_stage_statuses(run_dir: Path | None) -> tuple[dict[str, str], str | None]:
    """Infer per-stage statuses from artifact files and recent LLM activity."""
    statuses = {stage: "pending" for stage in _STAGE_NAMES}
    if run_dir is None:
        return statuses, None

    for stage in _STAGE_NAMES:
        if (run_dir / f"{stage}_output.json").exists():
            statuses[stage] = "done"

    active_stage = _read_last_llm_agent(run_dir)
    if active_stage not in statuses:
        active_stage = None

    if active_stage is None:
        for stage in _STAGE_NAMES:
            if statuses[stage] != "done":
                active_stage = stage
                break

    if active_stage is not None and statuses.get(active_stage) != "done":
        statuses[active_stage] = "running"

    return statuses, active_stage


def _render_live_stage_table(
    *,
    label: str,
    elapsed_seconds: float,
    statuses: dict[str, str],
    active_stage: str | None,
) -> Table:
    """Build the live per-stage status table shown during generation."""
    table = Table(
        show_header=True,
        header_style="bold",
        pad_edge=False,
        padding=(0, 1),
        title=f"{label} ({_format_duration(elapsed_seconds)})",
    )
    table.add_column("Stage", min_width=22)
    table.add_column("Status", min_width=12)

    for stage in _STAGE_NAMES:
        status = statuses.get(stage, "pending")
        if status == "done":
            status_text = "[green]done[/green]"
        elif status == "running":
            status_text = "[yellow]running[/yellow]"
        else:
            status_text = "[dim]pending[/dim]"
        stage_label = stage
        if stage == active_stage and status != "done":
            stage_label = f"{stage} [dim](active)[/dim]"
        table.add_row(stage_label, status_text)

    return table


def _run_with_stage_progress[T](label: str, fn: Callable[[], T]) -> tuple[T, float]:
    """Run StoryMesh with a live per-stage table and elapsed-time heartbeat."""
    result_box: dict[str, Any] = {}
    error_box: dict[str, BaseException] = {}
    store = ArtifactStore()

    def _target() -> None:
        try:
            result_box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001
            error_box["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    started_at = time.time()
    t0 = time.perf_counter()
    thread.start()

    with Live(console=console, refresh_per_second=8, transient=True) as live:
        while thread.is_alive():
            run_dir = _find_active_run_dir(store, started_at)
            statuses, active_stage = _infer_stage_statuses(run_dir)
            elapsed = time.perf_counter() - t0
            live.update(
                _render_live_stage_table(
                    label=label,
                    elapsed_seconds=elapsed,
                    statuses=statuses,
                    active_stage=active_stage,
                )
            )
            thread.join(timeout=0.125)

    elapsed = time.perf_counter() - t0
    if "error" in error_box:
        raise error_box["error"]
    return result_box["value"], elapsed


def _load_llm_usage_summary(
    store: ArtifactStore,
    run_id: str,
    *,
    include_agents: set[str] | None = None,
    exclude_agents: set[str] | None = None,
) -> dict[str, int]:
    """Summarize rough LLM usage from ``llm_calls.jsonl`` for one run."""
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


def _render_usage_line(label: str, usage: dict[str, int]) -> str:
    """Format one compact CLI line for rough LLM usage."""
    latency = _format_duration(usage["latency_ms"] / 1000) if usage["latency_ms"] else "0.0s"
    return (
        f"{label}: {usage['calls']} call(s), "
        f"~{usage['approx_prompt_tokens']:,} prompt / "
        f"~{usage['approx_response_tokens']:,} response / "
        f"~{usage['approx_total_tokens']:,} total tokens, "
        f"{latency} model time"
    )


@app.command()
def generate(
    user_prompt: str = typer.Argument(
        ...,
        help="Describe the fiction you want a synopsis for (genres, tones, setting, etc.).",
    ),
    quality: Annotated[
        str,
        typer.Option(
            "--quality",
            "-q",
            help="Quality preset: 'draft' (fast, no editorial cycle), "
            "'standard' (1 mandatory revision), 'high' (1 mandatory, threshold=8), "
            "'very_high' (2 mandatory revisions, threshold=9).",
        ),
    ] = "standard",
    prompt_style: Annotated[
        str | None,
        typer.Option(
            "--prompt-style",
            help="Prompt style to use. Defaults to the configured prompts.style value.",
        ),
    ] = None,
) -> None:
    """Generate an original fiction synopsis from the given prompt."""
    if quality not in _QUALITY_PRESETS:
        valid = ", ".join(sorted(_QUALITY_PRESETS))
        console.print(
            f"[bold red]Error:[/bold red] Unknown quality preset {quality!r}. "
            f"Valid options: {valid}"
        )
        raise typer.Exit(code=1)

    pass_threshold, max_retries, min_retries, enable_resonance = _QUALITY_PRESETS[quality]
    result, wall_clock = _run_with_stage_progress(
        "StoryMesh pipeline",
        lambda: generate_synopsis(
            user_prompt,
            pass_threshold=pass_threshold,
            max_retries=max_retries,
            min_retries=min_retries,
            skip_resonance_review=not enable_resonance,
            prompt_style=prompt_style,
        ),
    )

    meta = result.metadata
    run_id: str = str(meta.get("run_id", "unknown"))
    version: str = str(meta.get("pipeline_version", "?"))
    active_prompt_style: str = str(meta.get("prompt_style", "default"))
    stage_timings_raw = meta.get("stage_timings", {})
    stage_timings: dict[str, float] = stage_timings_raw if isinstance(stage_timings_raw, dict) else {}
    run_dir = Path(str(meta.get("run_dir", "")))

    # ── Header ────────────────────────────────────────────────────────────
    console.print(
        Panel(
            f"[bold]StoryMesh v{version}[/bold]  Run [dim]{run_id}[/dim]\n"
            f'Input: "[italic]{user_prompt}[/italic]"\n'
            f"Prompt style: [dim]{active_prompt_style}[/dim]",
            expand=False,
        )
    )
    console.print()

    # ── Stage table ───────────────────────────────────────────────────────
    table = Table(show_header=True, header_style="bold", pad_edge=False, padding=(0, 1))
    table.add_column("Stage", min_width=22)
    table.add_column("Status", min_width=8)
    table.add_column("Time", min_width=7)
    table.add_column("Artifact")

    for stage in _STAGE_NAMES:
        timing = stage_timings.get(stage, 0.0)
        artifact_file = run_dir / f"{stage}_output.json"
        if run_dir != Path("") and artifact_file.exists():
            status = "✓ done"
            artifact_str = str(artifact_file)
        else:
            status = "○ noop"
            artifact_str = "—"
        table.add_row(stage, status, f"{timing:.2f}s", artifact_str)

    total_time = sum(stage_timings.values())
    table.add_section()
    table.add_row("", f"Total: {total_time:.2f}s", "", "")

    console.print(table)
    console.print()

    # ── Synopsis ──────────────────────────────────────────────────────────
    console.print(Panel(result.final_synopsis, title="Synopsis", expand=False))
    console.print()

    # ── Output file paths (PDF / EPUB) ─────────────────────────────────────
    pdf_path = str(meta.get("pdf_path", ""))
    epub_path = str(meta.get("epub_path", ""))
    if pdf_path or epub_path:
        if pdf_path:
            console.print(f"PDF:  [dim]{pdf_path}[/dim]")
        if epub_path:
            console.print(f"EPUB: [dim]{epub_path}[/dim]")
        console.print()

    if run_dir != Path("") and run_dir.exists():
        console.print(f"Artifacts saved to: [dim]{run_dir}[/dim]")
        store = ArtifactStore()
        usage = _load_llm_usage_summary(store, run_id)
        if usage["calls"]:
            console.print(_render_usage_line("LLM usage (approx)", usage))
        console.print(f"Wall clock: [dim]{_format_duration(wall_clock)}[/dim]")


@app.command()
def compare(
    user_prompt: str = typer.Argument(
        ...,
        help="Describe the fiction you want to compare StoryMesh against a single-call baseline for.",
    ),
    quality: Annotated[
        str,
        typer.Option(
            "--quality",
            "-q",
            help="Quality preset for the StoryMesh run: 'draft', 'standard', 'high', or 'very_high'.",
        ),
    ] = "standard",
    prompt_style: Annotated[
        str | None,
        typer.Option(
            "--prompt-style",
            help="Prompt style for the StoryMesh run. Defaults to the configured prompts.style value.",
        ),
    ] = None,
    baseline_provider: Annotated[
        str | None,
        typer.Option(
            "--baseline-provider",
            help="Override provider for the single-call baseline. Defaults to story_writer provider.",
        ),
    ] = None,
    baseline_model: Annotated[
        str | None,
        typer.Option(
            "--baseline-model",
            help="Override model for the single-call baseline. Defaults to story_writer model.",
        ),
    ] = None,
    baseline_temperature: Annotated[
        float | None,
        typer.Option(
            "--baseline-temperature",
            help="Override temperature for the single-call baseline. Defaults to story_writer draft_temperature.",
        ),
    ] = None,
    baseline_max_tokens: Annotated[
        int | None,
        typer.Option(
            "--baseline-max-tokens",
            help="Override max_tokens for the single-call baseline. Defaults to story_writer draft_max_tokens.",
        ),
    ] = None,
) -> None:
    """Run StoryMesh and a one-shot baseline for the same prompt."""
    if quality not in _QUALITY_PRESETS:
        valid = ", ".join(sorted(_QUALITY_PRESETS))
        console.print(
            f"[bold red]Error:[/bold red] Unknown quality preset {quality!r}. "
            f"Valid options: {valid}"
        )
        raise typer.Exit(code=1)

    pass_threshold, max_retries, min_retries, enable_resonance = _QUALITY_PRESETS[quality]
    result, storymesh_wall_clock = _run_with_stage_progress(
        "StoryMesh pipeline",
        lambda: generate_synopsis(
            user_prompt,
            pass_threshold=pass_threshold,
            max_retries=max_retries,
            min_retries=min_retries,
            skip_resonance_review=not enable_resonance,
            prompt_style=prompt_style,
        ),
    )

    meta = result.metadata
    run_id = str(meta.get("run_id", ""))
    if not run_id:
        console.print("[bold red]Error:[/bold red] StoryMesh run did not produce a run_id.")
        raise typer.Exit(code=1)

    store = ArtifactStore()
    try:
        story_output = _load_story_writer_output(store, run_id)
        storymesh_draft = str(story_output.get("full_draft", "")).strip()
        storymesh_word_count = int(story_output.get("word_count", 0))

        if not storymesh_draft or storymesh_word_count <= 0:
            raise RuntimeError(
                "StoryMesh run did not produce a usable story draft for comparison."
            )

        baseline, baseline_wall_clock = _run_with_live_status(
            "Running single-call baseline...",
            lambda: _run_single_call_baseline(
                store=store,
                run_id=run_id,
                user_prompt=user_prompt,
                target_words=storymesh_word_count,
                provider_override=baseline_provider,
                model_override=baseline_model,
                temperature_override=baseline_temperature,
                max_tokens_override=baseline_max_tokens,
            ),
        )
    except RuntimeError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    comparison_data = {
        "user_prompt": user_prompt,
        "quality": quality,
        "storymesh": {
            "run_id": run_id,
            "prompt_style": meta.get("prompt_style", "default"),
            "word_count": storymesh_word_count,
            "final_synopsis": result.final_synopsis,
        },
        "baseline": {
            "provider": baseline["provider"],
            "model": baseline["model"],
            "temperature": baseline["temperature"],
            "max_tokens": baseline["max_tokens"],
            "target_words": storymesh_word_count,
            "word_count": baseline["word_count"],
            "wall_clock_seconds": round(baseline_wall_clock, 3),
        },
    }
    store.save_run_file(run_id, "comparison.json", comparison_data)
    blinded_packet, answer_key = _build_blinded_eval_packet(
        run_id=run_id,
        user_prompt=user_prompt,
        storymesh_draft=storymesh_draft,
        storymesh_word_count=storymesh_word_count,
        baseline_draft=str(baseline["full_draft"]),
        baseline_word_count=int(baseline["word_count"]),
    )
    store.save_run_file(run_id, "blinded_eval_packet.json", blinded_packet)
    store.save_run_file(run_id, "blinded_eval_key.json", answer_key)

    run_dir = Path(str(meta.get("run_dir", "")))
    console.print(
        Panel(
            f"[bold]StoryMesh vs Single-Call Baseline[/bold]\n"
            f'Input: "[italic]{user_prompt}[/italic]"\n'
            f"Run: [dim]{run_id}[/dim]\n"
            f"StoryMesh words: [dim]{storymesh_word_count}[/dim]\n"
            f"Baseline words: [dim]{baseline['word_count']}[/dim]\n"
            f"Baseline model: [dim]{baseline['provider']} / {baseline['model']}[/dim]",
            expand=False,
        )
    )
    console.print()
    console.print(f"Comparison artifacts saved to: [dim]{run_dir}[/dim]")
    console.print(f"  - [dim]{run_dir / 'story_writer_output.json'}[/dim]")
    console.print(f"  - [dim]{run_dir / 'baseline_output.json'}[/dim]")
    console.print(f"  - [dim]{run_dir / 'comparison.json'}[/dim]")
    console.print(f"  - [dim]{run_dir / 'blinded_eval_packet.json'}[/dim]")
    console.print(f"  - [dim]{run_dir / 'blinded_eval_key.json'}[/dim]")
    storymesh_usage = _load_llm_usage_summary(
        store,
        run_id,
        exclude_agents={"compare_baseline"},
    )
    baseline_usage = _load_llm_usage_summary(
        store,
        run_id,
        include_agents={"compare_baseline"},
    )
    if storymesh_usage["calls"]:
        console.print(_render_usage_line("StoryMesh LLM usage (approx)", storymesh_usage))
    if baseline_usage["calls"]:
        console.print(_render_usage_line("Baseline LLM usage (approx)", baseline_usage))
    console.print(
        "Wall clock: "
        f"[dim]StoryMesh {_format_duration(storymesh_wall_clock)}[/dim], "
        f"[dim]baseline {_format_duration(baseline_wall_clock)}[/dim]"
    )


_RERUN_SUPPORTED_STAGES = {"cover_art", "book_assembler"}


def _load_story_writer_output(store: ArtifactStore, run_id: str) -> dict[str, Any]:
    """Load story_writer_output.json for a completed run."""
    raw = store.load_run_file(run_id, "story_writer_output.json")
    if raw is None:
        raise RuntimeError(
            f"No story_writer_output.json found for run {run_id!r}."
        )
    data = orjson.loads(raw)
    if not isinstance(data, dict):
        raise RuntimeError(
            f"story_writer_output.json for run {run_id!r} is not a JSON object."
        )
    return data


def _run_single_call_baseline(
    *,
    store: ArtifactStore,
    run_id: str,
    user_prompt: str,
    target_words: int,
    provider_override: str | None,
    model_override: str | None,
    temperature_override: float | None,
    max_tokens_override: int | None,
) -> dict[str, Any]:
    """Generate one baseline story via a single LLM call and persist it."""
    from storymesh.config import get_agent_config  # noqa: PLC0415
    from storymesh.orchestration.graph import _build_llm_client  # noqa: PLC0415

    story_writer_cfg = get_agent_config("story_writer")
    baseline_cfg = dict(story_writer_cfg)
    if provider_override is not None:
        baseline_cfg["provider"] = provider_override
    if model_override is not None:
        baseline_cfg["model"] = model_override

    temperature = (
        temperature_override
        if temperature_override is not None
        else float(story_writer_cfg.get("draft_temperature", story_writer_cfg.get("temperature", 0.8)))
    )
    max_tokens = (
        max_tokens_override
        if max_tokens_override is not None
        else int(story_writer_cfg.get("draft_max_tokens", story_writer_cfg.get("max_tokens", 8000)))
    )

    llm_client = _build_llm_client(
        baseline_cfg,
        agent_name="compare_baseline",
        artifact_store=store,
    )
    if llm_client is None:
        raise RuntimeError(
            "No LLM client available for the single-call baseline. "
            "Check provider configuration and API keys."
        )

    user_prompt_text = _BASELINE_USER_TEMPLATE.format(
        user_prompt=user_prompt,
        target_words=target_words,
    )

    token = current_run_id.set(run_id)
    try:
        response = llm_client.complete_json(
            user_prompt_text,
            system_prompt=_BASELINE_SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    finally:
        current_run_id.reset(token)

    full_draft = str(response.get("full_draft", "")).strip()
    if not full_draft:
        raise RuntimeError("Single-call baseline returned an empty draft.")

    baseline_output = {
        "provider": baseline_cfg.get("provider"),
        "model": baseline_cfg.get("model"),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "target_words": target_words,
        "full_draft": full_draft,
        "word_count": len(full_draft.split()),
        "system_prompt": _BASELINE_SYSTEM_PROMPT,
        "user_prompt": user_prompt_text,
    }
    store.save_run_file(run_id, "baseline_output.json", baseline_output)
    return baseline_output


@app.command()
def rerun(
    stage: str = typer.Argument(
        ...,
        help="Pipeline stage to re-run ('cover_art' or 'book_assembler').",
    ),
    run_id: str | None = typer.Argument(
        None,
        help="Run ID to target. Omit to use the most recent run.",
    ),
) -> None:
    """Re-run a single pipeline stage for a previous run."""
    if stage not in _RERUN_SUPPORTED_STAGES:
        supported = ", ".join(sorted(_RERUN_SUPPORTED_STAGES))
        console.print(
            f"[bold red]Error:[/bold red] Stage {stage!r} does not support rerun. "
            f"Supported stages: {supported}"
        )
        raise typer.Exit(code=1)

    if stage == "cover_art":
        from storymesh import regenerate_cover_art  # noqa: PLC0415

        try:
            image_path = regenerate_cover_art(run_id)
        except (RuntimeError, ValueError) as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc

        console.print(f"[green]Cover art regenerated:[/green] {image_path}")

    elif stage == "book_assembler":
        from storymesh import regenerate_book_assembler  # noqa: PLC0415

        try:
            pdf_path, epub_path = regenerate_book_assembler(run_id)
        except (RuntimeError, ValueError) as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc

        if pdf_path:
            console.print(f"[green]PDF regenerated:[/green] {pdf_path}")
        if epub_path:
            console.print(f"[green]EPUB regenerated:[/green] {epub_path}")
        if not pdf_path and not epub_path:
            console.print("[yellow]No output files were generated (check library installation).[/yellow]")


@app.command()
def show_version() -> None:
    """Show the version of StoryMesh."""
    typer.echo(f"StoryMesh version {storymesh_version}")
    typer.echo("Schema Versions:")
    for name, ver in SCHEMA_VERSIONS.items():
        typer.echo(f"  - {name}: {ver}")
    typer.echo("Agent Versions:")
    for name, ver in AGENT_VERSIONS.items():
        typer.echo(f"  - {name}: {ver}")

@app.command()
def show_config() -> None:
    """Show the resolved StoryMesh configuration."""
    import yaml

    from storymesh.config import find_config_file, get_config

    config_path = find_config_file()
    typer.echo(f"Config loaded from: {config_path}")
    typer.echo("---")
    typer.echo(yaml.dump(get_config(), default_flow_style=False, sort_keys=False))

@app.command()
def show_agent_config(
    agent_name: str = typer.Argument(
            ...,
            help="Agent name to lookup."
        )) -> None:
        """Show the resolved LLM configuration for a specific agent."""

        import yaml

        from storymesh.config import get_agent_config

        resolved = get_agent_config(agent_name)
        typer.echo(f"Resolved config for '{agent_name}':")
        typer.echo("---")
        typer.echo(yaml.dump(resolved, default_flow_style=False, sort_keys=False))

@app.command()
def purge_cache(
    stages_only: bool = typer.Option(
        False, "--stages-only", help="Only purge the stage artifact cache."
    ),
    api_only: bool = typer.Option(
        False, "--api-only", help="Only purge the API response cache."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Purge cached stage outputs and/or API response caches."""
    purge_stages = not api_only
    purge_api = not stages_only

    store = ArtifactStore()
    config = get_config()
    api_cache_root = Path(
        config.get("cache", {}).get("dir", "~/.cache/storymesh")
    ).expanduser()

    console.print("[bold yellow]The following will be permanently deleted:[/bold yellow]")
    if purge_stages:
        console.print(f"  Stage cache:  {store.stages_dir}")
    if purge_api:
        console.print(f"  API cache:    {api_cache_root}")

    if not yes:
        typer.confirm("Continue?", abort=True)

    if purge_stages:
        count = store.purge_stage_cache()
        console.print(f"Stage cache cleared: {count} file(s) removed.")

    if purge_api:
        if api_cache_root.exists():
            shutil.rmtree(api_cache_root)
            console.print(f"API cache cleared: {api_cache_root}")
        else:
            console.print("API cache directory does not exist, nothing to remove.")


@app.command()
def purge_runs(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Purge all per-run artifact directories."""
    store = ArtifactStore()

    console.print(
        "[bold yellow]This will permanently delete all run data at:[/bold yellow]\n"
        f"  {store.runs_dir}"
    )

    if not yes:
        typer.confirm("Continue?", abort=True)

    count = store.purge_runs()
    console.print(f"Run data cleared: {count} run(s) removed.")


@app.command()
def inspect_run(
    run_id: str | None = typer.Argument(
        None,
        help="Run ID to inspect. Omit to use the most recent run.",
    ),
    llm: str | None = typer.Option(
        None,
        "--llm",
        help="Show full prompts and responses. Pass an agent name or 'all'.",
    ),
    html: Path | None = typer.Option(  # noqa: B008
        None,
        "--html",
        help="Write a self-contained HTML report to this path.",
        writable=True,
    ),
) -> None:
    """Inspect a past run's workflow and results stage-by-stage."""
    store = ArtifactStore()
    inspector = RunInspector(store)

    try:
        report = inspector.load(run_id if run_id else "latest")
    except RunNotFoundError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    _render_run_header(report)
    _render_stage_table(report)
    _render_stage_details(report)
    _render_llm_summary(report)

    if llm:
        _render_llm_full_detail(report, agent_filter=llm)

    if html is not None:
        html_content = inspector.generate_html(report)
        html.write_text(html_content, encoding="utf-8")
        console.print(f"\nHTML report written to: [dim]{html}[/dim]")


# ── inspect-run rendering helpers ─────────────────────────────────────────────


def _render_run_header(report: RunInspection) -> None:
    """Print the run header panel."""
    meta = report.metadata
    if meta:
        header = (
            f"[bold]StoryMesh v{meta.pipeline_version}[/bold]  "
            f"Run [dim]{report.run_id}[/dim]\n"
            f'Prompt: "[italic]{meta.user_prompt}[/italic]"\n'
            f"Timestamp: {meta.timestamp}"
        )
    else:
        header = f"Run [dim]{report.run_id}[/dim]\n[dim]Metadata unavailable.[/dim]"

    console.print()
    console.print(Panel(header, title="Run Summary", expand=False))
    console.print()


def _render_stage_table(report: RunInspection) -> None:
    """Print the stage timeline table."""
    timings: dict[str, float] = (
        report.metadata.stage_timings if report.metadata else {}
    )

    table = Table(show_header=True, header_style="bold", pad_edge=False, padding=(0, 1))
    table.add_column("Stage", min_width=22)
    table.add_column("Status", min_width=10)
    table.add_column("Time", min_width=7)
    table.add_column("Artifact")

    for name, stage in report.stages.items():
        timing = timings.get(name, 0.0)
        time_str = f"{timing:.2f}s" if timing else "—"

        if stage.status == StageStatus.DONE:
            status_str = "[green]✓ done[/green]"
            artifact_str = str(stage.artifact_path)
        elif stage.status == StageStatus.CORRUPT:
            status_str = "[red]! corrupt[/red]"
            artifact_str = str(stage.artifact_path)
        else:
            status_str = "[dim]○ missing[/dim]"
            artifact_str = "—"

        table.add_row(name, status_str, time_str, artifact_str)

    total = sum(timings.values())
    if total:
        table.add_section()
        table.add_row("", f"Total: {total:.2f}s", "", "")

    console.print(table)
    console.print()


def _render_stage_details(report: RunInspection) -> None:
    """Print a detail panel for each completed stage."""
    for name, stage in report.stages.items():
        if stage.status != StageStatus.DONE:
            continue
        _render_stage_panel(name, stage)


def _render_stage_panel(name: str, stage: StageInspection) -> None:
    """Dispatch to the stage-specific Rich renderer."""
    data = stage.data or {}
    if name == "genre_normalizer":
        content = _rich_genre_normalizer(data)
    elif name == "book_fetcher":
        content = _rich_book_fetcher(data)
    elif name == "book_ranker":
        content = _rich_book_ranker(data)
    elif name == "theme_extractor":
        content = _rich_theme_extractor(data)
    elif name == "proposal_draft":
        content = _rich_proposal_draft(data)
    else:
        content = "[dim](no detail renderer for this stage)[/dim]"

    console.print(Panel(content, title=name, expand=False))
    console.print()


def _rich_genre_normalizer(data: dict[str, Any]) -> str:
    """Format genre_normalizer stage output for Rich."""
    genres = ", ".join(str(g) for g in _as_list(data.get("normalized_genres"))) or "—"
    subgenres = ", ".join(str(g) for g in _as_list(data.get("subgenres"))) or "—"
    tones = ", ".join(str(t) for t in _as_list(data.get("user_tones"))) or "—"
    inferred_raw = _as_list(data.get("inferred_genres"))
    inferred = ", ".join(
        str(g.get("canonical_genre", g)) if isinstance(g, dict) else str(g)
        for g in inferred_raw
    ) or "—"
    context = ", ".join(str(c) for c in _as_list(data.get("narrative_context"))) or "—"
    lines = [
        f"[bold]Genres:[/bold]            {genres}",
        f"[bold]Subgenres:[/bold]         {subgenres}",
        f"[bold]Tones:[/bold]             {tones}",
        f"[bold]Inferred genres:[/bold]   {inferred}",
        f"[bold]Narrative context:[/bold] {context}",
    ]
    if data.get("tone_override") and data.get("override_note"):
        lines.append(f"[yellow]Tone override:[/yellow] {data['override_note']}")
    return "\n".join(lines)


def _rich_book_fetcher(data: dict[str, Any]) -> str:
    """Format book_fetcher stage output for Rich."""
    queries = ", ".join(str(q) for q in _as_list(data.get("queries_executed"))) or "—"
    books = _as_list(data.get("books"))
    lines = [
        f"[bold]Queries:[/bold]       {queries}",
        f"[bold]Books fetched:[/bold] {len(books)}",
        "",
        "[bold]Top books (up to 5):[/bold]",
    ]
    for book in books[:5]:
        if not isinstance(book, dict):
            continue
        title = book.get("title", "—")
        authors = ", ".join(str(a) for a in _as_list(book.get("authors")))
        year = book.get("first_publish_year", "—")
        lines.append(f"  • {title} — {authors} ({year})")
    return "\n".join(lines)


def _rich_book_ranker(data: dict[str, Any]) -> str:
    """Format book_ranker stage output for Rich."""
    ranked = _as_list(data.get("ranked_books"))
    dropped = data.get("dropped_count", 0)
    lines = [
        f"[bold]Books below cutoff:[/bold] {dropped}",
        "",
        "[bold]Top ranked books (up to 5):[/bold]",
    ]
    for rb in ranked[:5]:
        if not isinstance(rb, dict):
            continue
        book = rb.get("book", {})
        if not isinstance(book, dict):
            continue
        rank = rb.get("rank", "?")
        title = book.get("title", "—")
        score = float(rb.get("composite_score", 0))
        bd = rb.get("score_breakdown", {})
        go = float(bd.get("genre_overlap", 0)) if isinstance(bd, dict) else 0.0
        re_ = float(bd.get("reader_engagement", 0)) if isinstance(bd, dict) else 0.0
        lines.append(
            f"  {rank}. {title}  "
            f"[dim]score={score:.3f}  overlap={go:.3f}  engagement={re_:.3f}[/dim]"
        )
    return "\n".join(lines)


def _rich_theme_extractor(data: dict[str, Any]) -> str:
    """Format theme_extractor stage output for Rich."""
    lines: list[str] = []

    clusters = _as_list(data.get("genre_clusters"))
    lines.append("[bold]Genre Clusters:[/bold]")
    for c in clusters:
        if not isinstance(c, dict):
            continue
        genre = c.get("genre", "—")
        n = len(_as_list(c.get("books")))
        lines.append(f"  • {genre} ({n} book{'s' if n != 1 else ''})")

    tensions = _as_list(data.get("tensions"))
    lines.append("")
    lines.append("[bold]Thematic Tensions:[/bold]")
    for t in tensions:
        if not isinstance(t, dict):
            continue
        tid = t.get("tension_id", "—")
        cq = str(t.get("creative_question", ""))
        intensity = float(t.get("intensity", 0))
        truncated = cq[:120] + "…" if len(cq) > 120 else cq
        lines.append(f"  {tid} [dim](intensity={intensity:.2f})[/dim]: {truncated}")

    seeds = _as_list(data.get("narrative_seeds"))
    lines.append("")
    lines.append("[bold]Narrative Seeds:[/bold]")
    for s in seeds:
        if not isinstance(s, dict):
            continue
        sid = s.get("seed_id", "—")
        concept = str(s.get("concept", ""))
        truncated = concept[:150] + "…" if len(concept) > 150 else concept
        lines.append(f"  {sid}: {truncated}")

    return "\n".join(lines)


def _rich_proposal_draft(data: dict[str, Any]) -> str:
    """Format proposal_draft stage output for Rich."""
    proposal = data.get("proposal", {})
    if not isinstance(proposal, dict):
        proposal = {}
    rationale = data.get("selection_rationale", {})
    if not isinstance(rationale, dict):
        rationale = {}
    debug = data.get("debug", {})
    if not isinstance(debug, dict):
        debug = {}

    title = str(proposal.get("title", "—"))
    protagonist = str(proposal.get("protagonist", "—"))
    setting = str(proposal.get("setting", "—"))
    tone = str(proposal.get("tone", "—"))
    genre_blend = (
        ", ".join(str(g) for g in _as_list(proposal.get("genre_blend"))) or "—"
    )
    thematic_thesis = str(proposal.get("thematic_thesis", "—"))
    plot_arc = str(proposal.get("plot_arc", "—"))
    key_scenes = _as_list(proposal.get("key_scenes"))

    sel_index = rationale.get("selected_index", "—")
    n_valid = debug.get("num_valid_candidates", "—")
    runner_up = rationale.get("runner_up_index")
    runner_up_str = f", runner-up: {runner_up}" if runner_up is not None else ""
    rationale_text = str(rationale.get("rationale", "—"))
    cliche_violations = _as_list(rationale.get("cliche_violations"))

    n_requested = debug.get("num_candidates_requested", "—")
    n_failures = debug.get("num_parse_failures", 0)
    draft_temp = debug.get("draft_temperature", "—")
    total_calls = debug.get("total_llm_calls", "—")

    lines: list[str] = [
        f"[bold]{title}[/bold]",
        f"[bold]Protagonist:[/bold]      {protagonist}",
        f"[bold]Setting:[/bold]          {setting}",
        f"[bold]Tone:[/bold]             {tone}",
        f"[bold]Genre blend:[/bold]      {genre_blend}",
        f"[bold]Thematic thesis:[/bold]  {thematic_thesis}",
        "",
        "[bold]Plot arc:[/bold]",
        f"  {plot_arc[:600] + '…' if len(plot_arc) > 600 else plot_arc}",
    ]

    if key_scenes:
        lines.append("")
        lines.append("[bold]Key scenes:[/bold]")
        for i, scene in enumerate(key_scenes, 1):
            s = str(scene)
            lines.append(f"  {i}. {s[:120] + '…' if len(s) > 120 else s}")

    rationale_truncated = (
        rationale_text[:400] + "…" if len(rationale_text) > 400 else rationale_text
    )
    lines += [
        "",
        "[bold]Selection:[/bold]",
        (
            f"  Candidate {sel_index} of {n_valid}{runner_up_str} "
            f"[dim](requested={n_requested}  failures={n_failures}  "
            f"calls={total_calls}  temp={draft_temp})[/dim]"
        ),
        f"  {rationale_truncated}",
    ]

    if cliche_violations:
        lines.append("  [yellow]Cliché violations:[/yellow]")
        for v in cliche_violations:
            lines.append(f"    • {v}")

    return "\n".join(lines)


def _render_llm_summary(report: RunInspection) -> None:
    """Print the LLM call summary table."""
    if not report.llm_calls:
        console.print("[dim]No LLM calls recorded for this run.[/dim]")
        console.print()
        return

    table = Table(
        show_header=True, header_style="bold", pad_edge=False, padding=(0, 1),
        title="LLM Calls",
    )
    table.add_column("Agent", min_width=20)
    table.add_column("Model", min_width=24)
    table.add_column("Attempt", min_width=7)
    table.add_column("Latency (ms)", min_width=12)
    table.add_column("Approx Tokens", min_width=13)
    table.add_column("Parse OK", min_width=8)

    for call in report.llm_calls:
        latency = f"{call.latency_ms:.0f}" if call.latency_ms is not None else "—"
        approx_tokens = (
            str(call.approx_total_tokens)
            if call.approx_total_tokens is not None
            else "—"
        )
        parse_ok = "[green]✓[/green]" if call.parse_success else "[red]✗[/red]"
        table.add_row(call.agent, call.model, str(call.attempt), latency, approx_tokens, parse_ok)

    console.print(table)
    console.print()


def _render_llm_full_detail(report: RunInspection, agent_filter: str) -> None:
    """Print full prompt and response panels for LLM calls matching the filter."""
    calls: list[LLMCallRecord] = (
        report.llm_calls
        if agent_filter == "all"
        else [c for c in report.llm_calls if c.agent == agent_filter]
    )

    if not calls:
        console.print(
            f"[dim]No LLM calls found for agent filter '{agent_filter}'.[/dim]"
        )
        return

    for call in calls:
        label = f"{call.agent} / attempt {call.attempt}"
        console.print(Panel(call.system_prompt, title=f"{label} — system prompt", expand=False))
        console.print(Panel(call.user_prompt, title=f"{label} — user prompt", expand=False))
        console.print(Panel(call.raw_response, title=f"{label} — response", expand=False))
        console.print()


def _as_list(value: object) -> list[Any]:
    """Return value as a list, or an empty list if it is not a list."""
    return value if isinstance(value, list) else []


# ---------------------------------------------------------------------------
# stylometrics — offline tic counter
# ---------------------------------------------------------------------------


@app.command()
def stylometrics(
    run_id: str | None = typer.Argument(
        None,
        help="Run ID to inspect. Use 'latest' or omit to use the most recent run.",
    ),
    all_runs: bool = typer.Option(
        False,
        "--all",
        help="Count tics across all runs in the store.",
    ),
    pretty: bool = typer.Option(
        False,
        "--pretty",
        help="Render a human-readable table instead of JSON.",
    ),
) -> None:
    """Count prose tics in a story draft.

    Reads the story_writer_output.json (or resonance_reviewer_output.json if
    present) for the specified run and reports tic frequencies.  Output is
    informational only — no pass/fail thresholds.
    """
    import json as _json

    from storymesh.exceptions import RunNotFoundError

    store = ArtifactStore()
    inspector = RunInspector(store)

    target_ids: list[str]
    if all_runs:
        target_ids = store.list_run_ids()
        if not target_ids:
            console.print("[yellow]No runs found.[/yellow]")
            raise typer.Exit(0)
    else:
        resolved = run_id or "latest"
        try:
            report = inspector.load(resolved)
        except RunNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        target_ids = [report.run_id]

    results: list[dict[str, Any]] = []
    for rid in target_ids:
        result = _stylometrics_for_run(store, rid)
        if result is not None:
            results.append(result)

    if not results:
        console.print("[yellow]No story_writer_output found for the specified run(s).[/yellow]")
        raise typer.Exit(0)

    if pretty:
        for r in results:
            _render_stylometrics_table(r)
    else:
        if len(results) == 1:
            typer.echo(_json.dumps(results[0], indent=2))
        else:
            typer.echo(_json.dumps(results, indent=2))


def _stylometrics_for_run(store: ArtifactStore, run_id: str) -> dict[str, Any] | None:
    """Load draft text and metadata for a run and return count_tics result."""
    from storymesh.diagnostics.stylometric_counter import count_tics

    # Prefer the resonance-reviewed draft when available (it's the final text).
    raw = store.load_run_file(run_id, "resonance_reviewer_output.json")
    draft_key = "revised_draft"
    if raw is None:
        raw = store.load_run_file(run_id, "story_writer_output.json")
        draft_key = "full_draft"
    if raw is None:
        return None

    import json as _json

    try:
        data = _json.loads(raw)
    except Exception:
        return None

    draft = data.get(draft_key, "")
    if not draft:
        return None

    # Resolve metadata for title and voice profile.
    meta_raw = store.load_run_file(run_id, "run_metadata.json")
    title = "unknown"
    voice_profile_id = "unknown"
    if meta_raw:
        try:
            meta = _json.loads(meta_raw)
            title = str(meta.get("title", "unknown"))
        except Exception:
            pass

    # Pull voice_profile_id from voice_profile_selector_output if available.
    vps_raw = store.load_run_file(run_id, "voice_profile_selector_output.json")
    if vps_raw:
        try:
            vps = _json.loads(vps_raw)
            voice_profile_id = str(vps.get("selected_profile_id", "unknown"))
        except Exception:
            pass

    result = count_tics(draft)
    return {
        "run_id": run_id,
        "story_title": title,
        "voice_profile": voice_profile_id,
        **result,
    }


def _render_stylometrics_table(r: dict[str, Any]) -> None:
    """Render a stylometrics result as a Rich table."""
    console.print(
        f"\n[bold]Run:[/bold] {r['run_id']}  "
        f"[bold]Title:[/bold] {r['story_title']}  "
        f"[bold]Profile:[/bold] {r['voice_profile']}  "
        f"[bold]Words:[/bold] {r['word_count']}"
    )
    table = Table("Tic", "Count / Value", "Per 1000 words", box=None)
    for tic_name, tic_data in r.get("tics", {}).items():
        if "count" in tic_data:
            table.add_row(tic_name, str(tic_data["count"]), str(tic_data["per_1000_words"]))
        else:
            table.add_row(tic_name, str(tic_data["value"]), "—")
    console.print(table)


if __name__ == "__main__":
    app()
