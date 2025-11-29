"""Tests for owl env docs command."""

import json
from unittest.mock import Mock, patch

import pytest

from owa.cli.env import app as env_app
from owa.core.documentation.validator import ComponentValidationResult, PluginValidationResult


@pytest.fixture
def mock_validator():
    validator = Mock()
    good = PluginValidationResult(
        plugin_name="good_plugin",
        documented=2,
        total=2,
        good_quality=2,
        skipped=0,
        components=[
            ComponentValidationResult("good_plugin/c1", "good", []),
            ComponentValidationResult("good_plugin/c2", "good", []),
        ],
    )
    poor = PluginValidationResult(
        plugin_name="poor_plugin",
        documented=1,
        total=2,
        good_quality=0,
        skipped=0,
        components=[
            ComponentValidationResult("poor_plugin/c1", "poor", ["Missing docstring"]),
            ComponentValidationResult("poor_plugin/c2", "acceptable", ["Missing examples"]),
        ],
    )
    validator.validate_all_plugins.return_value = {"good_plugin": good, "poor_plugin": poor}
    validator.validate_plugin.return_value = good
    return validator


def test_docs_help(cli_runner):
    result = cli_runner.invoke(env_app, ["docs", "--help"])
    assert result.exit_code == 0
    assert "--output-format" in result.stdout


@patch("owa.cli.env.docs.DocumentationValidator")
def test_docs_table_format(mock_cls, cli_runner, mock_validator):
    mock_cls.return_value = mock_validator
    result = cli_runner.invoke(env_app, ["docs"])
    assert result.exit_code == 1
    assert "good_plugin" in result.stdout
    assert "poor_plugin" in result.stdout


@patch("owa.cli.env.docs.DocumentationValidator")
def test_docs_json_format(mock_cls, cli_runner, mock_validator):
    mock_cls.return_value = mock_validator
    result = cli_runner.invoke(env_app, ["docs", "--output-format=json"])
    assert result.exit_code == 1
    data = json.loads(result.stdout)
    assert data["result"] == "FAIL"
    assert data["plugins"]["good_plugin"]["status"] == "pass"


@patch("owa.cli.env.docs.DocumentationValidator")
def test_docs_invalid_format(mock_cls, cli_runner, mock_validator):
    mock_cls.return_value = mock_validator
    result = cli_runner.invoke(env_app, ["docs", "--output-format=invalid"])
    assert result.exit_code == 2


@patch("owa.cli.env.docs.DocumentationValidator")
def test_docs_specific_plugin(mock_cls, cli_runner, mock_validator):
    mock_cls.return_value = mock_validator
    result = cli_runner.invoke(env_app, ["docs", "good_plugin"])
    assert result.exit_code == 0
    mock_validator.validate_plugin.assert_called_once_with("good_plugin")
