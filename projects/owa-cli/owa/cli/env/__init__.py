import typer
from rich.console import Console

from . import docs, generate_docs, list, quick, search, show, stats, validate

app = typer.Typer(help="Environment plugin management commands.")

# Core commands
app.command("list")(list.list_plugins)
app.command("show")(show.show_plugin)
app.command("validate")(validate.validate_plugin)

# Enhanced commands
app.command("search")(search.search_components)
app.command("stats")(stats.show_stats)
app.command("health")(stats.health_check)

# Documentation commands (OEP-0004)
app.command("docs")(docs.docs_command)
app.command("ecosystem-docs")(docs.ecosystem_docs_command)
app.command("generate-docs")(generate_docs.generate_docs_command)
app.command("validate-docs")(validate.validate_docs_command)

# Quick access commands (essential shortcuts only)
app.command("ls")(quick.ls)
app.command("find")(quick.find)
app.command("namespaces")(quick.namespaces)

# Create console for rich output
console = Console()
