import typer

from . import docs, list, search, stats, validate

app = typer.Typer(help="Environment plugin management commands.")

# Core commands (minimal, intuitive design)
app.command("list")(list.list_env)
app.command("search")(search.search_components)
app.command("validate")(validate.validate_plugin)
app.command("stats")(stats.show_stats)
app.command("docs")(docs.docs)
