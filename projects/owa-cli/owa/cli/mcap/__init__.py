import importlib
import platform

import typer

from . import cat, convert, info

app = typer.Typer(help="MCAP file management commands.")

app.command()(cat.cat)
app.command()(convert.convert)
app.command()(info.info)

# if Windows and both `owa.env.desktop` and `owa.env.gst` are installed, add `record` command
if importlib.util.find_spec("owa.ocap"):
    from owa.ocap import record

    app.command()(record)
