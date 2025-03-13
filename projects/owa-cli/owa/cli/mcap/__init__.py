import typer

from . import cat, convert, info, record

app = typer.Typer(help="MCAP file management commands.")

app.command()(cat.cat)
app.command()(convert.convert)
app.command()(info.info)
app.command()(record.record)
