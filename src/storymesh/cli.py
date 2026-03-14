import typer

from storymesh import generate_synopsis
from storymesh.versioning import AGENT_VERSIONS, SCHEMA_VERSIONS
from storymesh.versioning import __version__ as storymesh_version

app = typer.Typer()

@app.command()
def generate(
    genre: str = typer.Argument(..., 
           help="Fiction genre or genre list to generate a synopsis for."),
) -> None:
    """Generate an original fiction synopsis for the given genre."""
    result = generate_synopsis(genre)
    typer.echo(f"Generated Synopsis:\n{result.final_synopsis}")
    typer.echo(f"Metadata: {result.metadata}")

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
