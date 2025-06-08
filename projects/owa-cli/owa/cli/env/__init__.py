import typer
from rich.console import Console

from . import docs, list, messages, quick, search, show, stats, validate

app = typer.Typer(help="Environment plugin management commands.")

# Core commands
app.command("list")(list.list_plugins)
app.command("show")(show.show_plugin)
app.command("validate")(validate.validate_plugin)

# Documentation commands (OEP-0004)
app.command("validate-docs")(docs.validate_docs)
app.command("docs-stats")(docs.docs_stats)

# Message commands (OEP-0005)
app.command("messages")(messages.list_messages)
app.command("show-message")(messages.show_message)
app.command("validate-messages")(messages.validate_messages)

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
