import typer
from rich.console import Console
from rich.tree import Tree

from . import list, show, validate

app = typer.Typer(help="Environment plugin management commands.")

# Add commands
app.command("list")(list.list_plugins)
app.command("show")(show.show_plugin)
app.command("validate")(validate.validate_plugin)

# Create console for rich output
console = Console()
