"""
Minimal, focused tests for ocap utility functions.

Only tests functions with actual business logic worth testing.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def setup_mocks():
    """Mock external dependencies to allow importing the module."""
    from unittest.mock import MagicMock

    # Mock only what's needed to import
    sys.modules["typer"] = MagicMock()
    sys.modules["loguru"] = MagicMock()
    sys.modules["tqdm"] = MagicMock()
    sys.modules["typing_extensions"] = MagicMock()
    sys.modules["mcap_owa"] = MagicMock()
    sys.modules["mcap_owa.highlevel"] = MagicMock()
    sys.modules["owa.core"] = MagicMock()
    sys.modules["owa.core.time"] = MagicMock()


class TestParseAdditionalProperties:
    """Test the parse_additional_properties function - the only parsing logic in the module."""

    def test_none_input(self):
        """None input should return empty dict."""
        setup_mocks()
        from owa.ocap.record import parse_additional_properties

        result = parse_additional_properties(None)
        assert result == {}

    def test_single_property(self):
        """Single key=value should parse correctly."""
        setup_mocks()
        from owa.ocap.record import parse_additional_properties

        result = parse_additional_properties("key=value")
        assert result == {"key": "value"}

    def test_multiple_properties(self):
        """Multiple properties should parse correctly."""
        setup_mocks()
        from owa.ocap.record import parse_additional_properties

        result = parse_additional_properties("key1=value1,key2=value2,key3=value3")
        assert result == {"key1": "value1", "key2": "value2", "key3": "value3"}

    def test_special_characters(self):
        """Properties with special characters should work."""
        setup_mocks()
        from owa.ocap.record import parse_additional_properties

        result = parse_additional_properties("path=/some/path,name=test-file,count=123")
        assert result == {"path": "/some/path", "name": "test-file", "count": "123"}

    def test_empty_string_reveals_bug(self):
        """Empty string reveals a bug in the original function."""
        setup_mocks()
        from owa.ocap.record import parse_additional_properties

        # This reveals that the function doesn't handle empty strings properly
        with pytest.raises(ValueError):
            parse_additional_properties("")


class TestEnsureOutputFilesReady:
    """Test the ensure_output_files_ready function - the only file handling logic."""

    @pytest.fixture
    def temp_dir(self):
        """Provide a temporary directory."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_basic_path_handling(self, temp_dir):
        """Test basic file path handling."""
        setup_mocks()
        from owa.ocap.record import ensure_output_files_ready

        output_path = temp_dir / "output"

        # Mock typer.confirm to avoid user interaction
        with patch("typer.confirm", return_value=True):
            result = ensure_output_files_ready(output_path)

        # Should return MCAP file path
        assert result == output_path.with_suffix(".mcap")
        assert result.suffix == ".mcap"

    def test_directory_creation(self, temp_dir):
        """Test that directories are created when needed."""
        setup_mocks()
        from owa.ocap.record import ensure_output_files_ready

        # Use nested path that doesn't exist
        output_path = temp_dir / "new_dir" / "output"

        with patch("typer.confirm", return_value=True):
            result = ensure_output_files_ready(output_path)

        # Directory should be created
        assert result.parent.exists()
        assert result.parent.is_dir()


# That's it! No need to test:
# - enqueue_event() - trivial one-liner
# - keyboard_monitor_callback() - simple logging + enqueue
# - screen_capture_callback() - simple path manipulation + enqueue
# - check_plugin() - single assertion, better tested via integration
# - setup_resources() - complex context manager, better tested end-to-end
# - record() - main CLI function, better tested via CLI integration
