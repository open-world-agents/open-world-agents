import typer

from . import probe, transcode, vfr_to_cfr

app = typer.Typer(help="Video processing commands.")

app.command("probe")(probe.analyze_video)
app.command("transcode")(transcode.main)
app.command("vfr-to-cfr")(vfr_to_cfr.main)
