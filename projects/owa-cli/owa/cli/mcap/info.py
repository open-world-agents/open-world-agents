import os
import platform
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import List

import requests
import typer
from packaging.version import parse as parse_version
from typing_extensions import Annotated

from mcap_owa.highlevel import OWAMcapReader

from ..console import console

# Template URLs for mcap CLI downloads - {version} will be replaced with actual version
MCAP_CLI_DOWNLOAD_URL_TEMPLATES = {
    "linux-amd64": "https://github.com/foxglove/mcap/releases/download/releases%2Fmcap-cli%2F{version}/mcap-linux-amd64",
    "linux-arm64": "https://github.com/foxglove/mcap/releases/download/releases%2Fmcap-cli%2F{version}/mcap-linux-arm64",
    "darwin-amd64": "https://github.com/foxglove/mcap/releases/download/releases%2Fmcap-cli%2F{version}/mcap-macos-amd64",
    "darwin-arm64": "https://github.com/foxglove/mcap/releases/download/releases%2Fmcap-cli%2F{version}/mcap-macos-arm64",
    "windows-amd64": "https://github.com/foxglove/mcap/releases/download/releases%2Fmcap-cli%2F{version}/mcap-windows-amd64.exe",
}
# Current version as fallback
CURRENT_MCAP_CLI_VERSION = "v0.0.54"


def get_mcap_info(file_path: Path) -> dict:
    """Get MCAP file information using OWAMcapReader."""
    with OWAMcapReader(file_path) as reader:
        duration_ns = reader.end_time - reader.start_time

        # Get MCAP file size
        mcap_size = file_path.stat().st_size

        # Check for corresponding .mkv file
        mkv_path = file_path.with_suffix(".mkv")
        mkv_size = 0
        has_mkv = False

        if mkv_path.exists():
            mkv_size = mkv_path.stat().st_size
            has_mkv = True

        return {
            "file_path": file_path,
            "messages": reader.message_count,
            "duration_seconds": duration_ns / 1e9,
            "size_bytes": mcap_size,
            "mkv_size_bytes": mkv_size,
            "total_size_bytes": mcap_size + mkv_size,
            "has_mkv": has_mkv,
            "channels": len(reader.topics),
        }


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m{remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}h{remaining_minutes}m{remaining_seconds:.1f}s"


def format_size(bytes_size: int) -> str:
    """Format file size in bytes to human-readable format."""
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.1f} KiB"
    elif bytes_size < 1024 * 1024 * 1024:
        return f"{bytes_size / (1024 * 1024):.1f} MiB"
    else:
        return f"{bytes_size / (1024 * 1024 * 1024):.1f} GiB"


def print_summary(file_infos: List[dict]) -> None:
    """Print summary of multiple MCAP files."""
    if not file_infos:
        console.print("No valid MCAP files found.")
        return

    # Calculate totals
    total_messages = sum(info["messages"] for info in file_infos)
    total_duration = sum(info["duration_seconds"] for info in file_infos)
    total_mcap_size = sum(info["size_bytes"] for info in file_infos)
    total_mkv_size = sum(info["mkv_size_bytes"] for info in file_infos)
    files_with_mkv = sum(1 for info in file_infos if info["has_mkv"])

    console.print(f"Summary for {len(file_infos)} MCAP files:")
    console.print(f"{'=' * 50}")
    console.print(f"Total messages:     {total_messages:,}")
    console.print(f"Total duration:     {format_duration(total_duration)}")
    console.print(f"Total MCAP size:    {format_size(total_mcap_size)}")
    if total_mkv_size > 0:
        console.print(f"Total MKV size:     {format_size(total_mkv_size)}")
        console.print(f"Files with MKV:     {files_with_mkv}/{len(file_infos)}")
    console.print(f"Files processed:    {len(file_infos)}")
    console.print()

    # Show per-file breakdown with limit to prevent stdout explosion
    MAX_FILES_TO_SHOW = 25  # Limit to prevent overwhelming output

    if len(file_infos) <= MAX_FILES_TO_SHOW:
        console.print("Per-file breakdown:")
    else:
        console.print(f"Per-file breakdown (showing first {MAX_FILES_TO_SHOW} of {len(file_infos)} files):")

    console.print(
        f"{'File':<25} {'Messages':<10} {'Duration':<12} {'MCAP Size':<12} {'Total Size':<12} {'MKV':<5} {'Ch':<4}"
    )
    console.print(f"{'-' * 25} {'-' * 10} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 5} {'-' * 4}")

    # Only show up to MAX_FILES_TO_SHOW files to prevent stdout explosion
    files_to_show = file_infos[:MAX_FILES_TO_SHOW]

    for info in files_to_show:
        filename = info["file_path"].name
        if len(filename) > 23:
            filename = filename[:20] + "..."

        # Format sizes
        mcap_size_str = format_size(info["size_bytes"])
        total_size_str = format_size(info["total_size_bytes"])
        mkv_indicator = "✓" if info["has_mkv"] else "✗"

        console.print(
            f"{filename:<25} {info['messages']:<10,} {format_duration(info['duration_seconds']):<12} "
            f"{mcap_size_str:<12} {total_size_str:<12} {mkv_indicator:<5} {info['channels']:<4}"
        )

    # Show indication if there are more files
    if len(file_infos) > MAX_FILES_TO_SHOW:
        remaining_files = len(file_infos) - MAX_FILES_TO_SHOW
        console.print(f"... and {remaining_files} more files (use individual file info command for details)")


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

    finally:
        # Clean up temp file if it still exists
        if temp_file.exists():
            temp_file.unlink()


def info(
    mcap_paths: Annotated[List[Path], typer.Argument(help="Path(s) to the input .mcap file(s)")],
    force_upgrade: Annotated[
        bool, typer.Option("--force-upgrade", help="Force upgrade mcap CLI to latest version")
    ] = False,
):
    """Display information about the .mcap file(s). Shows detailed info for single file, summary for multiple files."""
    # Validate all files exist
    for mcap_path in mcap_paths:
        if not mcap_path.exists():
            raise FileNotFoundError(f"MCAP file not found: {mcap_path}")

    # Detect Conda environment and get its bin directory
    bin_dir = get_conda_bin_dir()

    # Download or upgrade `mcap` CLI if needed
    download_mcap_cli(bin_dir, force_upgrade)

    mcap_executable = bin_dir / ("mcap.exe" if os.name == "nt" else "mcap")

    if len(mcap_paths) == 1:
        # Single file: show detailed info (original behavior)
        result = subprocess.run([mcap_executable, "info", str(mcap_paths[0])], text=True, capture_output=True)

        if result.returncode == 0:
            console.print(result.stdout)
        else:
            print(f"Error running mcap CLI: {result.stderr}", file=sys.stderr)
            sys.exit(result.returncode)
    else:
        # Multiple files: show summary
        file_infos = []
        errors = []

        for mcap_path in mcap_paths:
            try:
                file_info = get_mcap_info(mcap_path)
                file_infos.append(file_info)
            except Exception as e:
                errors.append(f"Error processing {mcap_path}: {e}")

        # Print any errors
        if errors:
            print("Errors encountered:", file=sys.stderr)
            for error in errors:
                print(f"  {error}", file=sys.stderr)
            print(file=sys.stderr)

        # Print summary if we have any valid files
        if file_infos:
            print_summary(file_infos)
        else:
            print("No valid MCAP files could be processed.", file=sys.stderr)
            sys.exit(1)


# Example usage:
if __name__ == "__main__":
    test_path = Path("example.mcap")
    info([test_path])
