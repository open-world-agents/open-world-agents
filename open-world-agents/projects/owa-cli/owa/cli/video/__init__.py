import typer

from . import probe, vfr_to_cfr

app = typer.Typer(help="MKV management commands.")

app.command("probe")(probe.analyze_video)
app.command("vfr-to-cfr")(vfr_to_cfr.main)
