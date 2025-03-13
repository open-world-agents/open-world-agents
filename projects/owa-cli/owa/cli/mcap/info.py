import os
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path

# Mapping of OS/arch to corresponding download URLs
MCAP_CLI_DOWNLOAD_URLS = {
    "linux-amd64": "https://github.com/foxglove/mcap/releases/download/releases%2Fmcap-cli%2Fv0.0.51/mcap-linux-amd64",
    "linux-arm64": "https://github.com/foxglove/mcap/releases/download/releases%2Fmcap-cli%2Fv0.0.51/mcap-linux-arm64",
    "darwin-amd64": "https://github.com/foxglove/mcap/releases/download/releases%2Fmcap-cli%2Fv0.0.51/mcap-macos-amd64",
    "darwin-arm64": "https://github.com/foxglove/mcap/releases/download/releases%2Fmcap-cli%2Fv0.0.51/mcap-macos-arm64",
    "windows-amd64": "https://github.com/foxglove/mcap/releases/download/releases%2Fmcap-cli%2Fv0.0.51/mcap-windows-amd64.exe",
}


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


def get_conda_bin_dir() -> Path:
    """Return the bin directory of the active conda environment."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        raise RuntimeError("No active conda environment detected.")
    return Path(conda_prefix) / ("Scripts" if os.name == "nt" else "bin")


def download_mcap_cli(bin_dir: Path):
    """Download the `mcap` CLI executable if not already installed."""
    system_key = detect_system()
    download_url = MCAP_CLI_DOWNLOAD_URLS.get(system_key)

    if not download_url:
        raise RuntimeError(f"No mcap CLI available for {system_key}")

    mcap_executable = bin_dir / ("mcap.exe" if "windows" in system_key else "mcap")

    if mcap_executable.exists():
        return  # Already installed

    print(f"Downloading mcap CLI from {download_url}...")
    urllib.request.urlretrieve(download_url, mcap_executable)

    # Make the file executable on Unix-based systems
    if not system_key.startswith("windows"):
        mcap_executable.chmod(0o755)

    print(f"mcap CLI installed at {mcap_executable}")


def info(mcap_path: Path):
    """Display information about the .mcap file."""
    if not mcap_path.exists():
        raise FileNotFoundError(f"MCAP file not found: {mcap_path}")

    # Detect Conda environment and get its bin directory
    bin_dir = get_conda_bin_dir()

    # Download `mcap` CLI if needed
    download_mcap_cli(bin_dir)

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
