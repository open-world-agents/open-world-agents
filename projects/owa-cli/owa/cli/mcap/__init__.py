import importlib
import platform

import typer

from . import cat, convert, info

app = typer.Typer(help="MCAP file management commands.")

app.command()(cat.cat)
app.command()(convert.convert)
app.command()(info.info)

# if Windows and both `owa.env.desktop` and `owa.env.gst` are installed, add `record` command
if (
    platform.system() == "Windows"
    and importlib.util.find_spec("owa.env.desktop")
    and importlib.util.find_spec("owa.env.gst")
):
    from . import record

    app.command()(record.record)
