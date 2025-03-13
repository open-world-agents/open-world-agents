import pygetwindow as gw
import typer

app = typer.Typer()


@app.command()
def resize(window_name: str, width: int, height: int):
    """
    Resize a window identified by its title.

    :param window_name: The title of the window to be resized.
    :param width: The new width of the window.
    :param height: The new height of the window.
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
