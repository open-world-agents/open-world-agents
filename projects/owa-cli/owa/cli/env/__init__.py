import typer

from . import docs, list, quick, search, show, stats, validate

app = typer.Typer(help="Environment plugin management commands.")

# Core commands
app.command("list")(list.list_plugins)
app.command("show")(show.show_plugin)
app.command("validate")(validate.validate_plugin)

# Documentation commands (OEP-0004)
app.command("validate-docs")(docs.validate_docs)
app.command("docs-stats")(docs.docs_stats)

# Enhanced commands
app.command("search")(search.search_components)
app.command("stats")(stats.show_stats)
app.command("health")(stats.health_check)

# Quick access commands (essential shortcuts only)
app.command("ls")(quick.ls)
app.command("find")(quick.find)
app.command("namespaces")(quick.namespaces)
