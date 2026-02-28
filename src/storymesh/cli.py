import typer

from storymesh import generate_synopsis
from storymesh.version import __version__ as storymesh_version

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

if __name__ == "__main__":
    app()
