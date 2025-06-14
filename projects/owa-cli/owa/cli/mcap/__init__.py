import importlib

import typer

from . import cat, convert, convert_legacy, info

app = typer.Typer(help="MCAP file management commands.")

app.command()(cat.cat)
app.command()(convert.convert)
app.command("convert-legacy")(convert_legacy.convert_legacy_mcap)
app.command()(info.info)

# if Windows and both `owa.env.desktop` and `owa.env.gst` are installed, add `record` command
if importlib.util.find_spec("owa.ocap"):
    from owa.ocap import record

    app.command()(record)
