import requests
from packaging.version import parse as parse_version
from rich import print


def get_local_version():
    try:
        from importlib.metadata import version
    except ImportError:  # For Python <3.8
        from importlib_metadata import version

    try:
        __version__ = version("owa")
    except Exception:
        __version__ = "unknown"

    return __version__


def get_latest_release():
    url = "https://api.github.com/repos/open-world-agents/open-world-agents/releases/latest"
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    tag = response.json()["tag_name"]
    return tag.lstrip("v")  # Remove leading "v" if present


def check_for_update():
    local_version = get_local_version()
    latest_version = get_latest_release()
    if parse_version(latest_version) > parse_version(local_version):
        print(f"""
[bold red]******************************************************[/bold red]
[bold yellow]   An update is available for Open World Agents![/bold yellow]
[bold red]******************************************************[/bold red]
[bold]  Your version:[/bold] [red]{local_version}[/red]    [bold]Latest:[/bold] [green]{latest_version}[/green]
  Get it here: [bold cyan]https://github.com/open-world-agents/open-world-agents/releases[/bold cyan]
""")
