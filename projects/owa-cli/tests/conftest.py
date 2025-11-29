"""Pytest configuration for owa-cli tests."""

import os
import shutil
import warnings
from pathlib import Path

import pytest
from typer.testing import CliRunner


def pytest_configure(config):
    os.environ["OWA_DISABLE_CONSOLE_STYLING"] = "1"
    os.environ["OWA_DISABLE_VERSION_CHECK"] = "1"


@pytest.fixture
def cli_runner():
    # NOTE: env vars needed here too for GitHub Actions (not just pytest_configure)
    return CliRunner(
        charset="utf-8",
        env={"NO_COLOR": "1", "TERM": "dumb", "TTY_COMPATIBLE": "1", "TTY_INTERACTIVE": "0"},
    )


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent / "data"


@pytest.fixture
def copy_test_file():
    def _copy(source_dir: Path, filename: str, dest_dir: Path) -> Path:
        source = source_dir / filename
        dest = dest_dir / filename
        if source.exists():
            shutil.copy2(source, dest)
            return dest
        pytest.skip(f"Test file {filename} not found")

    return _copy


@pytest.fixture
def suppress_mcap_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Reader version.*", UserWarning)
        warnings.filterwarnings("ignore", "unclosed file.*", ResourceWarning)
        yield
