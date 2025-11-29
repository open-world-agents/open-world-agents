"""
Tests for the owl env validate command.
"""

import pytest
import yaml

from owa.cli.env import app as env_app
from owa.core.plugin_spec import PluginSpec


@pytest.fixture
def sample_yaml(tmp_path):
    """Create a temporary YAML file for testing."""
    yaml_file = tmp_path / "test_plugin.yaml"
    yaml_file.write_text(
        yaml.dump(
            {
                "namespace": "test_plugin",
                "version": "1.0.0",
                "description": "Test plugin",
                "components": {"callables": {"hello": "test.module:hello"}},
            }
        )
    )
    return str(yaml_file)


@pytest.fixture
def invalid_yaml(tmp_path):
    """Create a temporary invalid YAML file for testing."""
    yaml_file = tmp_path / "invalid.yaml"
    yaml_file.write_text(
        yaml.dump(
            {
                "namespace": "invalid_plugin",
                "version": "1.0.0",
                "description": "Invalid plugin",
                "components": {"callables": {"bad": "invalid_format_no_colon"}},
            }
        )
    )
    return str(yaml_file)


def test_validate_yaml_success(cli_runner, sample_yaml):
    """Test successful validation of a YAML file."""
    result = cli_runner.invoke(env_app, ["validate", sample_yaml, "--no-check-imports"])
    assert result.exit_code == 0
    assert "Plugin Specification Valid" in result.stdout


def test_validate_yaml_with_errors(cli_runner, invalid_yaml):
    """Test validation of a YAML file with import errors."""
    result = cli_runner.invoke(env_app, ["validate", invalid_yaml])
    assert result.exit_code == 1
    assert "missing ':'" in result.stdout


def test_validate_entry_point(cli_runner):
    """Test successful validation of an entry point."""
    result = cli_runner.invoke(env_app, ["validate", "owa.env.plugins.std:plugin_spec", "--no-check-imports"])
    assert result.exit_code == 0
    assert "std" in result.stdout


def test_validate_nonexistent(cli_runner):
    """Test validation of a non-existent YAML file."""
    result = cli_runner.invoke(env_app, ["validate", "nonexistent.yaml"])
    assert result.exit_code == 1


def test_plugin_spec_from_yaml(tmp_path):
    """Test PluginSpec.from_yaml method directly."""
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(
        yaml.dump(
            {
                "namespace": "test",
                "version": "1.0.0",
                "description": "Test",
                "components": {"callables": {"fn": "m:f"}},
            }
        )
    )
    spec = PluginSpec.from_yaml(str(yaml_file))
    assert spec.namespace == "test"
    assert spec.components["callables"]["fn"] == "m:f"


def test_plugin_spec_from_entry_point():
    """Test PluginSpec.from_entry_point method directly."""
    spec = PluginSpec.from_entry_point("owa.env.plugins.std:plugin_spec")
    assert spec.namespace == "std"
    assert "callables" in spec.components


def test_plugin_spec_invalid_format():
    """Test PluginSpec.from_entry_point with invalid format."""
    with pytest.raises(ValueError, match="Invalid entry point format"):
        PluginSpec.from_entry_point("invalid_format")


def test_validate_nonexistent_entry_point(cli_runner):
    """Test validation of a non-existent entry point."""
    result = cli_runner.invoke(env_app, ["validate", "nonexistent.module:plugin_spec"])
    assert result.exit_code == 1
    assert "Cannot import module" in result.stdout


def test_validate_verbose_mode(cli_runner, sample_yaml):
    """Test validation with verbose mode."""
    result = cli_runner.invoke(env_app, ["validate", sample_yaml, "--verbose", "--no-check-imports"])
    assert result.exit_code == 0
    assert "Detected input type: yaml" in result.stdout


def test_plugin_spec_from_entry_point_nonexistent_module():
    """Test PluginSpec.from_entry_point with non-existent module."""
    with pytest.raises(ImportError, match="Cannot import module"):
        PluginSpec.from_entry_point("nonexistent.module:plugin_spec")
