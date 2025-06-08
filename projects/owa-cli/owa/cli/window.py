import pygetwindow as gw
import typer
from typing_extensions import Annotated

from owa.core.registry import CALLABLES

app = typer.Typer(help="Window management commands.")


@app.command()
def find(window_name: str):
    """
    Find a window by its title.
    """
    window = CALLABLES["desktop/window.get_window_by_title"](window_name)
    height, width = window.rect[3] - window.rect[1], window.rect[2] - window.rect[0]

    typer.echo(f"Window Found: '{typer.style(window.title, fg=typer.colors.GREEN, bold=True)}'")
    typer.echo(f"├─ Dimensions: {typer.style(f'{width} × {height}', fg=typer.colors.YELLOW)} pixels")
    typer.echo(
        f"├─ Position: {typer.style(f'({window.rect[0]}, {window.rect[1]})', fg=typer.colors.CYAN)} to {typer.style(f'({window.rect[2]}, {window.rect[3]})', fg=typer.colors.CYAN)}"
    )
    typer.echo(f"├─ Handle (hWnd): {typer.style(str(window.hWnd), fg=typer.colors.MAGENTA)}")

    try:
        import win32process

        pid = win32process.GetWindowThreadProcessId(window.hWnd)[1]
        typer.echo(f"└─ Process ID: {typer.style(str(pid), fg=typer.colors.BLUE)}")
    except ImportError:
        typer.echo(
            f"└─ Process ID: {typer.style('Not available (win32process module required)', fg=typer.colors.RED)}"
        )


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
