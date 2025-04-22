import platform

import typer

from . import mcap, video
from .utils import check_for_update

# Check for updates on startup
check_for_update()

# Define the main Typer app
app = typer.Typer()
app.add_typer(mcap.app, name="mcap")
app.add_typer(video.app, name="video")

if platform.system() == "Windows":
    from . import window

    app.add_typer(window.app, name="window")
else:
    typer.echo("Since you're not using Windows OS, `owa-cli window` command is disabled.", err=True)
