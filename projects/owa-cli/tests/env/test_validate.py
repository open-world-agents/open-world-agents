"""
Tests for the owl env validate command.
"""

import tempfile
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from owa.cli.env import app as env_app
from owa.core.plugin_spec import PluginSpec


@pytest.fixture
def runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def sample_yaml_file():
    """Create a temporary YAML file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(
            {
                "namespace": "test_plugin",
                "version": "1.0.0",
                "description": "Test plugin for validation",
                "author": "Test Author",
                "components": {
                    "callables": {"hello": "test.module:hello_function", "add": "test.module:add_function"},
                    "listeners": {"events": "test.module:EventListener"},
                },
            },
            f,
        )
        yield f.name
    Path(f.name).unlink()


@pytest.fixture
def invalid_yaml_file():
    """Create a temporary invalid YAML file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(
            {
                "namespace": "invalid_plugin",
                "version": "1.0.0",
                "description": "Invalid plugin for testing",
                "components": {
                    "callables": {"bad_import": "invalid_format_no_colon", "good_import": "test.module:good_function"}
                },
            },
            f,
        )
        yield f.name
    Path(f.name).unlink()


def test_validate_yaml_file_success(runner, sample_yaml_file):
    """Test successful validation of a YAML file."""
    result = runner.invoke(env_app, ["validate", sample_yaml_file, "--no-check-imports"])
    assert result.exit_code == 0
    assert "✅ Plugin Specification Valid" in result.stdout
    assert "test_plugin" in result.stdout
    assert "YAML file:" in result.stdout


def test_validate_yaml_file_with_warnings(runner, invalid_yaml_file):
    """Test validation of a YAML file with import warnings."""
    result = runner.invoke(env_app, ["validate", invalid_yaml_file])
    assert result.exit_code == 0
    assert "⚠️  Plugin Specification Valid (with warnings)" in result.stdout
    assert "Import Validation Warnings" in result.stdout
    assert "missing ':'" in result.stdout


def test_validate_entry_point_success(runner):
    """Test successful validation of an entry point."""
    result = runner.invoke(env_app, ["validate", "owa.env.plugins.std:plugin_spec", "--no-check-imports"])
    assert result.exit_code == 0
    assert "✅ Plugin Specification Valid" in result.stdout
    assert "std" in result.stdout
    assert "Entry point:" in result.stdout


def test_validate_nonexistent_yaml_file(runner):
    """Test validation of a non-existent YAML file."""
    result = runner.invoke(env_app, ["validate", "nonexistent.yaml"])
    assert result.exit_code == 1
    assert "YAML file not found" in result.stdout


def test_validate_nonexistent_entry_point(runner):
    """Test validation of a non-existent entry point."""
    result = runner.invoke(env_app, ["validate", "nonexistent.module:plugin_spec"])
    assert result.exit_code == 1
    assert "Entry point validation failed" in result.stdout
    assert "Cannot import module" in result.stdout


def test_validate_verbose_mode(runner, sample_yaml_file):
    """Test validation with verbose mode."""
    result = runner.invoke(env_app, ["validate", sample_yaml_file, "--verbose", "--no-check-imports"])
    assert result.exit_code == 0
    assert "Detected input type: yaml" in result.stdout


def test_plugin_spec_from_yaml():
    """Test PluginSpec.from_yaml method directly."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(
            {
                "namespace": "direct_test",
                "version": "2.0.0",
                "description": "Direct test plugin",
                "components": {"callables": {"test": "test.module:test_function"}},
            },
            f,
        )
        f.flush()

        spec = PluginSpec.from_yaml(f.name)
        assert spec.namespace == "direct_test"
        assert spec.version == "2.0.0"
        assert spec.description == "Direct test plugin"
        assert "callables" in spec.components
        assert spec.components["callables"]["test"] == "test.module:test_function"

    Path(f.name).unlink()


def test_plugin_spec_from_entry_point():
    """Test PluginSpec.from_entry_point method directly."""
    spec = PluginSpec.from_entry_point("owa.env.plugins.std:plugin_spec")
    assert spec.namespace == "std"
    assert spec.version == "0.1.0"
    assert "callables" in spec.components


def test_plugin_spec_from_entry_point_invalid_format():
    """Test PluginSpec.from_entry_point with invalid format."""
    with pytest.raises(ValueError, match="Invalid entry point format"):
        PluginSpec.from_entry_point("invalid_format")


def test_plugin_spec_from_entry_point_nonexistent_module():
    """Test PluginSpec.from_entry_point with non-existent module."""
    with pytest.raises(ImportError, match="Cannot import module"):
        PluginSpec.from_entry_point("nonexistent.module:plugin_spec")
