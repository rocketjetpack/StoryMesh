import shutil
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
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
    result = generate_synopsis(
        user_prompt,
        pass_threshold=pass_threshold,
        max_retries=max_retries,
        min_retries=min_retries,
        skip_resonance_review=not enable_resonance,
    )

    meta = result.metadata
    run_id: str = str(meta.get("run_id", "unknown"))
    version: str = str(meta.get("pipeline_version", "?"))
    stage_timings_raw = meta.get("stage_timings", {})
    stage_timings: dict[str, float] = stage_timings_raw if isinstance(stage_timings_raw, dict) else {}
    run_dir = Path(str(meta.get("run_dir", "")))

    # ── Header ────────────────────────────────────────────────────────────
    console.print(
        Panel(
            f"[bold]StoryMesh v{version}[/bold]  Run [dim]{run_id}[/dim]\n"
            f'Input: "[italic]{user_prompt}[/italic]"',
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


_RERUN_SUPPORTED_STAGES = {"cover_art", "book_assembler"}


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
    table.add_column("Parse OK", min_width=8)

    for call in report.llm_calls:
        latency = f"{call.latency_ms:.0f}" if call.latency_ms is not None else "—"
        parse_ok = "[green]✓[/green]" if call.parse_success else "[red]✗[/red]"
        table.add_row(call.agent, call.model, str(call.attempt), latency, parse_ok)

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


if __name__ == "__main__":
    app()
