import importlib

import typer

from . import cat, info, migrate, rename_uri, sanitize, subtitle, trim

app = typer.Typer(help="MCAP file management commands.")

app.command()(cat.cat)
app.command()(subtitle.subtitle)
app.add_typer(migrate.app, name="migrate")
app.command()(info.info)
app.command(name="rename-uri")(rename_uri.rename_uri)
app.command()(sanitize.sanitize)
app.command()(trim.trim)

# if Windows and both `owa.env.desktop` and `owa.env.gst` are installed, add `record` command
if importlib.util.find_spec("owa.ocap"):
    from owa.ocap import record

    app.command()(record)
