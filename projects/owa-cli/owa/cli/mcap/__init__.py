import importlib

import typer

from . import cat, convert, info, migrate, sanitize

app = typer.Typer(help="MCAP file management commands.")

app.command()(cat.cat)
app.command()(convert.convert)
app.command()(migrate.migrate)
app.command()(info.info)
app.command()(sanitize.sanitize)

# if Windows and both `owa.env.desktop` and `owa.env.gst` are installed, add `record` command
if importlib.util.find_spec("owa.ocap"):
    from owa.ocap import record

    app.command()(record)
