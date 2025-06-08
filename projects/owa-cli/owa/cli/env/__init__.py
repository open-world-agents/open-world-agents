import typer
from rich.console import Console

from . import list, quick, search, show, stats, validate

app = typer.Typer(help="Environment plugin management commands.")

# Core commands
app.command("list")(list.list_plugins)
app.command("show")(show.show_plugin)
app.command("validate")(validate.validate_plugin)

# Enhanced commands
app.command("search")(search.search_components)
app.command("stats")(stats.show_stats)
app.command("health")(stats.health_check)

# Quick access commands (essential shortcuts only)
app.command("ls")(quick.ls)
app.command("find")(quick.find)
app.command("namespaces")(quick.namespaces)

# Create console for rich output
console = Console()
