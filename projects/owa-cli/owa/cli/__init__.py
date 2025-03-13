import platform

import typer

from . import mcap

app = typer.Typer()
app.add_typer(mcap.app, name="mcap")


if platform.system() == "Windows":
    from . import window

    app.add_typer(window.app, name="window")
else:
    typer.echo("Since you're not using Windows OS, `owa-cli window` command is disabled.")
