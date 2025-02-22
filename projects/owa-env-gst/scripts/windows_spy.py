import typer

from owa_env_desktop.window.callables import get_window_by_title


def main(window_title_substring: str):
    window = get_window_by_title(window_title_substring)
    print(window)


if __name__ == "__main__":
    typer.run(main)
