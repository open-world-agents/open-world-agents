"""
Tests for the unified owl env docs command.
"""

import json
from unittest.mock import Mock, patch

import pytest

from owa.core.documentation.validator import ComponentValidationResult, PluginValidationResult


@pytest.fixture
def mock_validator():
    """Create a mock DocumentationValidator."""
    validator = Mock()

    # Create mock validation results with correct constructor parameters
    good_result = PluginValidationResult(
        plugin_name="good_plugin",
        documented=2,
        total=2,
        good_quality=2,
        skipped=0,
        components=[
            ComponentValidationResult("good_plugin/component1", "good", []),
            ComponentValidationResult("good_plugin/component2", "good", []),
        ],
    )

    poor_result = PluginValidationResult(
        plugin_name="poor_plugin",
        documented=1,
        total=2,
        good_quality=0,
        skipped=0,
        components=[
            ComponentValidationResult("poor_plugin/component1", "poor", ["Missing docstring"]),
            ComponentValidationResult("poor_plugin/component2", "acceptable", ["Missing examples"]),
        ],
    )

    validator.validate_all_plugins.return_value = {"good_plugin": good_result, "poor_plugin": poor_result}

    validator.validate_plugin.return_value = good_result

    return validator


class TestUnifiedDocsCommand:
    """Test the unified docs command functionality."""

    def test_docs_help(self, cli_runner, env_app):
        """Test docs command help shows unified interface."""
        result = cli_runner.invoke(env_app, ["docs", "--help"])
        assert result.exit_code == 0
        assert "Validate plugin documentation quality" in result.stdout
        assert "--output-format" in result.stdout
        assert "table or json" in result.stdout

    @patch("owa.cli.env.docs.DocumentationValidator")
    def test_docs_table_format_default(self, mock_validator_class, cli_runner, mock_validator, env_app):
        """Test docs command with table format (default)."""
        mock_validator_class.return_value = mock_validator

        result = cli_runner.invoke(env_app, ["docs"])
        assert result.exit_code == 1  # Should fail due to poor_plugin
        assert "Documentation Statistics" in result.stdout
        assert "good_plugin" in result.stdout
        assert "poor_plugin" in result.stdout
        assert "Overall Coverage" in result.stdout
        assert "Overall Result: FAIL" in result.stdout
        assert "Improvements needed" in result.stdout

    @patch("owa.cli.env.docs.DocumentationValidator")
    def test_docs_json_format(self, mock_validator_class, cli_runner, mock_validator, env_app):
        """Test docs command with JSON format."""
        mock_validator_class.return_value = mock_validator

        result = cli_runner.invoke(env_app, ["docs", "--output-format=json"])
        assert result.exit_code == 1  # Should fail due to poor_plugin

        # Parse JSON output
        output_data = json.loads(result.stdout)
        assert output_data["result"] == "FAIL"
        assert "good_plugin" in output_data["plugins"]
        assert "poor_plugin" in output_data["plugins"]
        assert output_data["plugins"]["good_plugin"]["status"] == "pass"

    @patch("owa.cli.env.docs.DocumentationValidator")
    def test_docs_invalid_format(self, mock_validator_class, cli_runner, mock_validator, env_app):
        """Test docs command with invalid format."""
        mock_validator_class.return_value = mock_validator

        result = cli_runner.invoke(env_app, ["docs", "--output-format=invalid"])
        assert result.exit_code == 2
        assert "Invalid format 'invalid'" in result.stdout
        assert "Must be 'table' or 'json'" in result.stdout

    @patch("owa.cli.env.docs.DocumentationValidator")
    def test_docs_by_type_flag(self, mock_validator_class, cli_runner, mock_validator, env_app):
        """Test docs command with by-type flag."""
        mock_validator_class.return_value = mock_validator

        result = cli_runner.invoke(env_app, ["docs", "--by-type"])
        assert result.exit_code == 1
        assert "Documentation Statistics by Type" in result.stdout
        assert "by-type view" in result.stdout

    @patch("owa.cli.env.docs.DocumentationValidator")
    def test_docs_specific_plugin(self, mock_validator_class, cli_runner, mock_validator, env_app):
        """Test docs command with specific plugin."""
        mock_validator_class.return_value = mock_validator

        result = cli_runner.invoke(env_app, ["docs", "good_plugin"])
        assert result.exit_code == 0  # Should pass for good_plugin
        mock_validator.validate_plugin.assert_called_once_with("good_plugin")

    @patch("owa.cli.env.docs.DocumentationValidator")
    def test_docs_strict_mode(self, mock_validator_class, cli_runner, mock_validator, env_app):
        """Test docs command with strict mode."""
        mock_validator_class.return_value = mock_validator

        result = cli_runner.invoke(env_app, ["docs", "--strict"])
        assert result.exit_code == 1  # Should fail in strict mode
        mock_validator.validate_all_plugins.assert_called_once()

    @patch("owa.cli.env.docs.DocumentationValidator")
    def test_docs_exit_codes(self, mock_validator_class, cli_runner, env_app):
        """Test proper exit codes."""
        # Test success case
        mock_validator = Mock()
        good_result = PluginValidationResult(
            plugin_name="good_plugin",
            documented=1,
            total=1,
            good_quality=1,
            skipped=0,
            components=[ComponentValidationResult("good_plugin/component1", "good", [])],
        )
        mock_validator.validate_all_plugins.return_value = {"good_plugin": good_result}
        mock_validator_class.return_value = mock_validator

        result = cli_runner.invoke(env_app, ["docs"])
        assert result.exit_code == 0

        # Test failure case
        poor_result = PluginValidationResult(
            plugin_name="poor_plugin",
            documented=0,
            total=1,
            good_quality=0,
            skipped=0,
            components=[ComponentValidationResult("poor_plugin/component1", "poor", ["Missing docstring"])],
        )
        mock_validator.validate_all_plugins.return_value = {"poor_plugin": poor_result}

        result = cli_runner.invoke(env_app, ["docs"])
        assert result.exit_code == 1
