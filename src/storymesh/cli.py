from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from storymesh import generate_synopsis
from storymesh.versioning import AGENT_VERSIONS, SCHEMA_VERSIONS
from storymesh.versioning import __version__ as storymesh_version

app = typer.Typer()
console = Console()

# Pipeline stage names in execution order — used to build the stage table.
_STAGE_NAMES = [
    "genre_normalizer",
    "book_fetcher",
    "book_ranker",
    "theme_extractor",
    "proposal_draft",
    "rubric_judge",
    "synopsis_writer",
]


@app.command()
def generate(
    user_prompt: str = typer.Argument(
        ...,
        help="Describe the fiction you want a synopsis for (genres, tones, setting, etc.).",
    ),
) -> None:
    """Generate an original fiction synopsis from the given prompt."""
    result = generate_synopsis(user_prompt)

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

    if run_dir != Path("") and run_dir.exists():
        console.print(f"Artifacts saved to: [dim]{run_dir}[/dim]")


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

if __name__ == "__main__":
    app()
