import os
import platform
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import requests
import typer
from packaging.version import parse as parse_version
from typing_extensions import Annotated

# Template URLs for mcap CLI downloads - {version} will be replaced with actual version
MCAP_CLI_DOWNLOAD_URL_TEMPLATES = {
    "linux-amd64": "https://github.com/foxglove/mcap/releases/download/releases%2Fmcap-cli%2F{version}/mcap-linux-amd64",
    "linux-arm64": "https://github.com/foxglove/mcap/releases/download/releases%2Fmcap-cli%2F{version}/mcap-linux-arm64",
    "darwin-amd64": "https://github.com/foxglove/mcap/releases/download/releases%2Fmcap-cli%2F{version}/mcap-macos-amd64",
    "darwin-arm64": "https://github.com/foxglove/mcap/releases/download/releases%2Fmcap-cli%2F{version}/mcap-macos-arm64",
    "windows-amd64": "https://github.com/foxglove/mcap/releases/download/releases%2Fmcap-cli%2F{version}/mcap-windows-amd64.exe",
}
# Current version as fallback
CURRENT_MCAP_CLI_VERSION = "v0.0.53"


def detect_system():
    """Detect OS and architecture to determine which mcap binary to use."""
    system_os = platform.system().lower()
    arch = platform.machine().lower()

    if system_os == "linux":
        os_key = "linux"
    elif system_os == "darwin":
        os_key = "darwin"
    elif system_os == "windows":
        os_key = "windows"
    else:
        raise RuntimeError(f"Unsupported OS: {system_os}")

    # Standardize architecture name
    if "arm" in arch or "aarch64" in arch:
        arch_key = "arm64"
    elif "x86_64" in arch or "amd64" in arch:
        arch_key = "amd64"
    else:
        raise RuntimeError(f"Unsupported architecture: {arch}")

    return f"{os_key}-{arch_key}"


def get_local_mcap_version(mcap_executable: Path) -> str:
    """Get the version of the local mcap CLI binary."""
    try:
        result = subprocess.run([mcap_executable, "version"], text=True, capture_output=True, timeout=10)
        if result.returncode == 0:
            # Parse version from output like "v0.0.53"
            version = result.stdout.strip()
            if version.startswith("v") and re.match(r"v\d+\.\d+\.\d+", version):
                return version
        return "unknown"
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return "unknown"


def get_latest_mcap_cli_version() -> str:
    """Get the latest mcap CLI version from GitHub releases."""
    # Skip GitHub API call if disabled via environment variable (e.g., during testing)
    if os.environ.get("OWA_DISABLE_VERSION_CHECK"):
        return CURRENT_MCAP_CLI_VERSION

    try:
        url = "https://api.github.com/repos/foxglove/mcap/releases"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        releases = response.json()
        # Find the latest mcap-cli release
        for release in releases:
            tag_name = release.get("tag_name", "")
            if tag_name.startswith("releases/mcap-cli/"):
                # Extract version from tag like "releases/mcap-cli/v0.0.53"
                version = tag_name.split("/")[-1]
                return version

        return CURRENT_MCAP_CLI_VERSION  # Fallback to current version
    except (requests.RequestException, Exception):
        return CURRENT_MCAP_CLI_VERSION  # Fallback to current version


def should_upgrade_mcap(mcap_executable: Path, force: bool = False) -> bool:
    """Check if mcap CLI should be upgraded."""
    if force:
        return True

    if not mcap_executable.exists():
        return True  # Need to download

    local_version = get_local_mcap_version(mcap_executable)
    latest_version = get_latest_mcap_cli_version()

    if local_version == "unknown":
        return True  # Can't determine version, safer to upgrade

    try:
        return parse_version(latest_version.lstrip("v")) > parse_version(local_version.lstrip("v"))
    except Exception:
        return False  # If version parsing fails, don't upgrade


def get_conda_bin_dir() -> Path:
    """Return the bin directory of the active conda environment."""
    conda_prefix = os.environ.get("CONDA_PREFIX") or os.environ.get("VIRTUAL_ENV")
    if not conda_prefix:
        raise RuntimeError("No active conda environment detected.")
    return Path(conda_prefix) / ("Scripts" if os.name == "nt" else "bin")


def download_mcap_cli(bin_dir: Path, force_upgrade: bool = False):
    """Download or upgrade the `mcap` CLI executable."""
    system_key = detect_system()
    mcap_executable = bin_dir / ("mcap.exe" if "windows" in system_key else "mcap")

    # Check if upgrade is needed
    if not should_upgrade_mcap(mcap_executable, force_upgrade):
        return  # Already up to date

    # Get the latest version and format URLs
    latest_version = get_latest_mcap_cli_version()

    # Format download URL with the latest version
    url_template = MCAP_CLI_DOWNLOAD_URL_TEMPLATES.get(system_key)
    if not url_template:
        raise RuntimeError(f"No mcap CLI available for {system_key}")

    download_url = url_template.format(version=latest_version)

    # Show appropriate message
    if mcap_executable.exists():
        local_version = get_local_mcap_version(mcap_executable)
        print(f"Upgrading mcap CLI from {local_version} to {latest_version}...")
    else:
        print(f"Downloading mcap CLI {latest_version}...")

    print(f"Downloading from {download_url}...")

    # Download to temporary file first
    temp_file = mcap_executable.with_suffix(mcap_executable.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(download_url, temp_file)

        # Make the file executable on Unix-based systems
        if not system_key.startswith("windows"):
            temp_file.chmod(0o755)

        # Replace the old file with the new one
        if mcap_executable.exists():
            mcap_executable.unlink()
        temp_file.rename(mcap_executable)

        print(f"mcap CLI {latest_version} installed at {mcap_executable}")

    except Exception as e:
        # Clean up temp file if download failed
        if temp_file.exists():
            temp_file.unlink()
        raise RuntimeError(f"Failed to download mcap CLI: {e}")


def info(
    mcap_path: Annotated[Path, typer.Argument(help="Path to the input .mcap file")],
    force_upgrade: Annotated[
        bool, typer.Option("--force-upgrade", help="Force upgrade mcap CLI to latest version")
    ] = False,
):
    """Display information about the .mcap file."""
    if not mcap_path.exists():
        raise FileNotFoundError(f"MCAP file not found: {mcap_path}")

    # Detect Conda environment and get its bin directory
    bin_dir = get_conda_bin_dir()

    # Download or upgrade `mcap` CLI if needed
    download_mcap_cli(bin_dir, force_upgrade)

    # Run `mcap info <mcap_path>`
    mcap_executable = bin_dir / ("mcap.exe" if os.name == "nt" else "mcap")
    result = subprocess.run([mcap_executable, "info", str(mcap_path)], text=True, capture_output=True)

    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"Error running mcap CLI: {result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)


# Example usage:
if __name__ == "__main__":
    test_path = Path("example.mcap")
    info(test_path)
