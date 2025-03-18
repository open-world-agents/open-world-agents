import pygetwindow as gw
import typer
from typing_extensions import Annotated

from owa.core.registry import CALLABLES, activate_module

app = typer.Typer(help="Window management commands.")


@app.command()
def find(window_name: str):
    """
    Find a window by its title.
    """
    activate_module("owa.env.desktop")
    window = CALLABLES["window.get_window_by_title"](window_name)
    typer.echo(window)


@app.command()
def resize(
    window_name: Annotated[str, typer.Argument(help="The title of the window to be resized.")],
    width: Annotated[int, typer.Argument(help="The new width of the window.")],
    height: Annotated[int, typer.Argument(help="The new height of the window.")],
):
    """
    Resize a window identified by its title.
    """
    try:
        # Attempt to find the window
        window = gw.getWindowsWithTitle(window_name)

        if not window:
            typer.echo(f"Error: No window found with the name '{window_name}'")
            raise typer.Exit(1)

        # Resize the first matching window
        win = window[0]
        win.resizeTo(width, height)
        typer.echo(f"Successfully resized '{window_name}' to {width}x{height}")

    except Exception as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
