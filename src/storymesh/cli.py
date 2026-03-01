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

if __name__ == "__main__":
    app()
