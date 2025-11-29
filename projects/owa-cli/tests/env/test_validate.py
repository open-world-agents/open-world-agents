"""Tests for owl env validate command."""

import pytest
import yaml

from owa.cli.env import app as env_app
from owa.core.plugin_spec import PluginSpec


@pytest.fixture
def sample_yaml(tmp_path):
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
    result = cli_runner.invoke(env_app, ["validate", sample_yaml, "--no-check-imports"])
    assert result.exit_code == 0
    assert "Plugin Specification Valid" in result.stdout


def test_validate_yaml_with_errors(cli_runner, invalid_yaml):
    result = cli_runner.invoke(env_app, ["validate", invalid_yaml])
    assert result.exit_code == 1
    assert "missing ':'" in result.stdout


def test_validate_entry_point(cli_runner):
    result = cli_runner.invoke(env_app, ["validate", "owa.env.plugins.std:plugin_spec", "--no-check-imports"])
    assert result.exit_code == 0
    assert "std" in result.stdout


def test_validate_nonexistent(cli_runner):
    result = cli_runner.invoke(env_app, ["validate", "nonexistent.yaml"])
    assert result.exit_code == 1


def test_plugin_spec_from_yaml(tmp_path):
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
    spec = PluginSpec.from_entry_point("owa.env.plugins.std:plugin_spec")
    assert spec.namespace == "std"
    assert "callables" in spec.components


def test_plugin_spec_invalid_format():
    with pytest.raises(ValueError, match="Invalid entry point format"):
        PluginSpec.from_entry_point("invalid_format")


def test_validate_nonexistent_entry_point(cli_runner):
    result = cli_runner.invoke(env_app, ["validate", "nonexistent.module:plugin_spec"])
    assert result.exit_code == 1
    assert "Cannot import module" in result.stdout
