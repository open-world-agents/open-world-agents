import importlib
import platform
import shutil

import typer

from . import mcap, video
from .utils import check_for_update

# Check for updates on startup
check_for_update()

# Define the main Typer app
app = typer.Typer()
app.add_typer(mcap.app, name="mcap")

if shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None:
    app.add_typer(video.app, name="video")
else:
    typer.echo("FFmpeg and/or ffprobe are not installed. `owa-cli video` command is disabled.", err=True)

if platform.system() == "Windows" and importlib.util.find_spec("owa.env.desktop"):
    from . import window

    app.add_typer(window.app, name="window")
else:
    if platform.system() != "Windows":
        typer.echo("`owa-cli window` command is disabled: not running on Windows OS.", err=True)
    elif not importlib.util.find_spec("owa.env.desktop"):
        typer.echo("`owa-cli window` command is disabled: 'owa.env.desktop' module is not installed.", err=True)
