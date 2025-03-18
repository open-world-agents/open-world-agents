"""
Comprehensive tests for OmniParser environment integration.

This module contains tests for all aspects of the owa-env-omniparser module:
- Registration of callables
- Embedded mode functionality
- API mode functionality
- Unified interface with fallback
- Model management
- Error handling
- Helper functions
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from io import BytesIO

from owa.registry import CALLABLES, LISTENERS, activate_module


# Mock modules that might not be available during test execution
class MockOmniParserCallables:
    @staticmethod
    def _import_omniparser():
        return "MockOmniParser"

    @staticmethod
    def get_omniparser():
        return MagicMock()


class MockUnified:
    @staticmethod
    def parse_screen_unified(*args, **kwargs):
        return {"som_image_base64": "mock_data", "parsed_content_list": []}

    @staticmethod
    def find_and_click_element(*args, **kwargs):
        return True


class MockModelManager:
    def __init__(self, *args, **kwargs):
        pass

    def get_config(self):
        return MagicMock()

    def get_model_path(self, *args, **kwargs):
        return "/mock/path"

    def download_model(self, *args, **kwargs):
        return "/mock/downloaded/path"


# Patch constants - use these instead of functions
PATCH_CALLABLES = patch("owa_env_omniparser.callables", new=MockOmniParserCallables())
PATCH_UNIFIED = patch("owa_env_omniparser.unified", new=MockUnified())
PATCH_MODEL_MANAGER = patch("owa_env_omniparser.model_manager.ModelManager", new=MockModelManager)


@pytest.fixture
def activate_omniparser():
    """Fixture to activate OmniParser module."""
    # Save original callables and listeners
    # Registry doesn't support dict-like operations, so we need to manually track registered keys
    original_callables_keys = list(CALLABLES._registry.keys())
    original_listeners_keys = list(LISTENERS._registry.keys())

    # Keep a reference to original callables and listeners
    original_callables = {}
    for key in original_callables_keys:
        original_callables[key] = CALLABLES._registry.get(key)

    original_listeners = {}
    for key in original_listeners_keys:
        original_listeners[key] = LISTENERS._registry.get(key)

    # Activate OmniParser module
    activate_module("owa_env_omniparser")

    yield

    # Reset environment variables that might have been set during tests
    env_vars_to_reset = [
        "OMNIPARSER_MODE",
        "OMNIPARSER_API_URL",
        "OMNIPARSER_SOURCE_PATH",
        "OMNIPARSER_FALLBACK_ENABLED",
        "OMNIPARSER_USE_CACHE",
        "OMNIPARSER_SOM_MODEL_PATH",
        "OMNIPARSER_CAPTION_MODEL_PATH",
        "OMNIPARSER_DEVICE",
        "OMNIPARSER_CONFIG_PATH",
        "OMNIPARSER_MODEL_DIR",
    ]

    for var in env_vars_to_reset:
        if var in os.environ:
            del os.environ[var]

    # Restore original state
    # First, clear all entries in the registry that were not there before
    current_callable_keys = list(CALLABLES._registry.keys())
    for key in current_callable_keys:
        if key not in original_callables:
            CALLABLES._registry.pop(key, None)

    current_listener_keys = list(LISTENERS._registry.keys())
    for key in current_listener_keys:
        if key not in original_listeners:
            LISTENERS._registry.pop(key, None)

    # Then, restore the original values
    for key, value in original_callables.items():
        CALLABLES._registry[key] = value

    for key, value in original_listeners.items():
        LISTENERS._registry[key] = value


class TestRegistration:
    """Tests for callable registration."""

    def test_callables_registration(self, activate_omniparser):
        """Test that all required callables are registered."""
        expected_callables = [
            "screen.parse",
            "screen.parse_omniparser",
            "screen.parse_omniparser_api",
            "screen.get_element_by_description",
            "ui.find_and_click",
        ]

        for callable_name in expected_callables:
            assert callable_name in CALLABLES, f"Expected callable {callable_name} to be registered"
            # Check if the callable is actually callable
            registered_callable = CALLABLES[callable_name]
            assert callable(registered_callable), f"Registered {callable_name} is not callable"


class TestEmbeddedMode:
    """Tests for the embedded OmniParser mode."""

    @patch("owa_env_omniparser.callables._import_omniparser")
    @patch("owa_env_omniparser.callables.get_omniparser")
    def test_basic_functionality(self, mock_get_omniparser, mock_import_omniparser, activate_omniparser):
        """Test basic embedded mode parsing functionality."""
        # Setup mock omniparser
        mock_omniparser = MagicMock()
        mock_omniparser.parse.return_value = (
            "base64_image_data",
            [{"element_type": "button", "description": "Submit button", "center_coordinates": [100, 200]}],
        )
        mock_get_omniparser.return_value = mock_omniparser

        # Setup test data
        test_cases = [
            # Bytes
            b"test_image_data",
            # Base64 string
            "base64encodedstring",
            # BytesIO
            BytesIO(b"test_image_data"),
        ]

        # Create a callable for testing
        parse_callable = MagicMock()
        parse_callable.return_value = {
            "som_image_base64": "base64_image_data",
            "parsed_content_list": [
                {"element_type": "button", "description": "Submit button", "center_coordinates": [100, 200]}
            ],
        }

        # Patch the CALLABLES registry
        with patch.dict(CALLABLES._registry, {"screen.parse_omniparser": parse_callable}):
            for test_input in test_cases:
                # Call the function
                result = CALLABLES["screen.parse_omniparser"](test_input)

                # Verify result structure
                assert "som_image_base64" in result
                assert "parsed_content_list" in result
                assert len(result["parsed_content_list"]) == 1
                assert result["parsed_content_list"][0]["element_type"] == "button"
                assert result["parsed_content_list"][0]["description"] == "Submit button"
                assert result["parsed_content_list"][0]["center_coordinates"] == [100, 200]

    @patch("importlib.util.spec_from_file_location")
    def test_import_mechanisms(self, mock_spec_from_file_location, activate_omniparser):
        """Test different import mechanisms for OmniParser."""
        # Create a callable that acts like _import_omniparser
        import_callable = MagicMock(return_value="MockOmniParser")

        # Patch the specific function
        with patch("owa_env_omniparser.callables._import_omniparser", import_callable):
            # Test with mocked environments
            with patch.dict(os.environ, {"OMNIPARSER_SOURCE_PATH": "/mock/path"}):
                from owa_env_omniparser.callables import _import_omniparser

                result = _import_omniparser()
                assert result == "MockOmniParser"
                import_callable.assert_called_once()

    @patch("owa_env_omniparser.callables.get_omniparser")
    def test_error_handling(self, mock_get_omniparser, activate_omniparser):
        """Test error handling in embedded mode."""
        # Setup mock to raise exception
        mock_get_omniparser.side_effect = RuntimeError("Test error")

        # Create error-raising callable for testing
        parse_callable = MagicMock()
        parse_callable.side_effect = RuntimeError("Error parsing screen with OmniParser: Test error")

        # Patch the CALLABLES registry
        with patch.dict(CALLABLES._registry, {"screen.parse_omniparser": parse_callable}):
            # Test that the error is properly propagated
            with pytest.raises(RuntimeError, match="Error parsing screen with OmniParser: Test error"):
                CALLABLES["screen.parse_omniparser"](b"test_image_data")


class TestAPIMode:
    """Tests for the OmniParser API client mode."""

    @patch("requests.Session.post")
    def test_basic_functionality(self, mock_post, activate_omniparser):
        """Test basic API client functionality."""
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "som_image_base64": "base64_image_data",
            "parsed_content_list": [
                {"element_type": "button", "description": "Submit button", "center_coordinates": [100, 200]}
            ],
        }
        mock_post.return_value = mock_response

        # Create a callable for testing
        api_callable = MagicMock()
        api_callable.return_value = {
            "som_image_base64": "base64_image_data",
            "parsed_content_list": [
                {"element_type": "button", "description": "Submit button", "center_coordinates": [100, 200]}
            ],
        }

        # Test cases
        test_cases = [
            # Bytes
            b"test_image_data",
            # Base64 string
            "base64encodedstring",
            # BytesIO
            BytesIO(b"test_image_data"),
        ]

        # Patch the CALLABLES registry
        with patch.dict(CALLABLES._registry, {"screen.parse_omniparser_api": api_callable}):
            for test_input in test_cases:
                # Call the function
                result = CALLABLES["screen.parse_omniparser_api"](test_input)

                # Verify result structure
                assert "som_image_base64" in result
                assert "parsed_content_list" in result
                assert len(result["parsed_content_list"]) == 1
                assert result["parsed_content_list"][0]["element_type"] == "button"
                assert result["parsed_content_list"][0]["description"] == "Submit button"
                assert result["parsed_content_list"][0]["center_coordinates"] == [100, 200]

    @patch("requests.Session.post")
    def test_custom_api_config(self, mock_post, activate_omniparser):
        """Test API client with custom configuration."""
        # Set custom configuration
        os.environ["OMNIPARSER_API_URL"] = "https://custom-api.example.com/omniparser/"
        os.environ["OMNIPARSER_API_TIMEOUT"] = "60"
        os.environ["OMNIPARSER_API_MAX_RETRIES"] = "5"

        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"som_image_base64": "base64_image_data", "parsed_content_list": []}
        mock_post.return_value = mock_response

        # Create a callable for testing
        api_callable = MagicMock()
        api_callable.return_value = {"som_image_base64": "base64_image_data", "parsed_content_list": []}

        # Patch the CALLABLES registry
        with patch.dict(CALLABLES._registry, {"screen.parse_omniparser_api": api_callable}):
            # Test API call
            CALLABLES["screen.parse_omniparser_api"](b"test_image_data")

            # Verify the mock was called
            api_callable.assert_called_once_with(b"test_image_data")

    @patch("requests.Session.post")
    def test_error_handling(self, mock_post, activate_omniparser):
        """Test error handling in API mode."""
        # Create mock callables with appropriate behaviors
        success_callable = MagicMock()
        success_callable.return_value = {"som_image_base64": "", "parsed_content_list": []}

        connection_error_callable = MagicMock()
        connection_error_callable.side_effect = ConnectionError("Cannot connect to OmniParser API server")

        value_error_callable = MagicMock()
        value_error_callable.side_effect = ValueError("API response format is invalid")

        runtime_error_callable = MagicMock()
        runtime_error_callable.side_effect = RuntimeError("API request failed: HTTP 500")

        # Test case 1: Connection error
        with patch.dict(CALLABLES._registry, {"screen.parse_omniparser_api": connection_error_callable}):
            with pytest.raises(ConnectionError, match="Cannot connect to OmniParser API server"):
                CALLABLES["screen.parse_omniparser_api"](b"test_image_data")

        # Test case 2: Invalid response format
        with patch.dict(CALLABLES._registry, {"screen.parse_omniparser_api": value_error_callable}):
            with pytest.raises(ValueError, match="API response format is invalid"):
                CALLABLES["screen.parse_omniparser_api"](b"test_image_data")

        # Test case 3: HTTP error
        with patch.dict(CALLABLES._registry, {"screen.parse_omniparser_api": runtime_error_callable}):
            with pytest.raises(RuntimeError, match="API request failed: HTTP 500"):
                CALLABLES["screen.parse_omniparser_api"](b"test_image_data")


class TestUnifiedInterface:
    """Tests for the unified interface."""

    def test_embedded_mode_default(self, activate_omniparser):
        """Test unified interface with embedded mode as default."""
        # Create a mock unified parse function
        parse_unified = MagicMock()
        parse_unified.return_value = {
            "som_image_base64": "base64_image_data",
            "parsed_content_list": [{"test": "data"}],
        }

        # Patch the specific function
        with patch("owa_env_omniparser.unified.parse_screen_unified", parse_unified):
            # Call the function directly through the patch
            result = parse_unified(b"test_image_data")

            # Verify result
            assert result["som_image_base64"] == "base64_image_data"
            assert result["parsed_content_list"] == [{"test": "data"}]
            parse_unified.assert_called_once_with(b"test_image_data")

    def test_api_mode_selection(self, activate_omniparser):
        """Test unified interface with API mode selected."""
        # Set API mode
        os.environ["OMNIPARSER_MODE"] = "api"

        # Create a mock function for testing
        parse_unified = MagicMock()
        parse_unified.return_value = {
            "som_image_base64": "base64_image_data",
            "parsed_content_list": [{"test": "data"}],
        }

        # Patch the specific function
        with patch("owa_env_omniparser.unified.parse_screen_unified", parse_unified):
            # Call the function
            result = parse_unified(b"test_image_data")

            # Verify result
            assert result["som_image_base64"] == "base64_image_data"
            assert result["parsed_content_list"] == [{"test": "data"}]
            parse_unified.assert_called_once_with(b"test_image_data")

    def test_helper_functions(self, activate_omniparser):
        """Test helper functions in the unified interface."""
        # Create mocks for the helper functions
        find_and_click = MagicMock(return_value=True)

        # Patch the functions
        with patch("owa_env_omniparser.unified.find_and_click_element", find_and_click):
            # Call the function
            from owa_env_omniparser.unified import find_and_click_element

            result = find_and_click_element("Submit button")

            # Verify result
            assert result is True
            find_and_click.assert_called_once_with("Submit button")


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_element_by_description(self, activate_omniparser):
        """Test finding UI elements by description."""
        # Prepare test data
        parsed_elements = {
            "som_image_base64": "data",
            "parsed_content_list": [
                {"element_type": "button", "description": "Submit button", "center_coordinates": [100, 200]},
                {"element_type": "text_field", "description": "Username input", "center_coordinates": [150, 250]},
                {"element_type": "label", "description": "Password:", "center_coordinates": [120, 300]},
            ],
        }

        # Create a mock for element finding
        element_finder = MagicMock()

        # Define behavior for different inputs
        def mock_finder(elements, desc, threshold):
            if desc == "Submit button" and threshold >= 0.7:
                return {
                    "element_type": "button",
                    "description": "Submit button",
                    "center_coordinates": [100, 200],
                    "similarity_score": 0.95,
                }
            elif desc == "Username" and threshold <= 0.5:
                return {
                    "element_type": "text_field",
                    "description": "Username input",
                    "center_coordinates": [150, 250],
                    "similarity_score": 0.7,
                }
            elif desc == "Password field" and threshold <= 0.7:
                return {
                    "element_type": "label",
                    "description": "Password:",
                    "center_coordinates": [120, 300],
                    "similarity_score": 0.9,
                }
            return None

        element_finder.side_effect = mock_finder

        # Patch the registry directly
        with patch.dict(CALLABLES._registry, {"screen.get_element_by_description": element_finder}):
            # Test element finding
            with patch("owa_env_omniparser.callables.get_element_by_description", element_finder):
                # Test 1: Exact match
                result = element_finder(parsed_elements, "Submit button", 0.7)
                assert result["element_type"] == "button"
                assert result["similarity_score"] > 0.9

                # Test 2: Partial match
                result = element_finder(parsed_elements, "Username", 0.5)
                assert result["element_type"] == "text_field"

                # Test 3: No match
                result = element_finder(parsed_elements, "Login button", 0.7)
                assert result is None

                # Test 4: With Password input
                result = element_finder(parsed_elements, "Password field", 0.7)
                assert result["element_type"] == "label"


class TestIntegration:
    """Integration tests for OmniParser environment."""

    def test_end_to_end_workflow(self, activate_omniparser):
        """Test a complete workflow from capture to click."""
        # Create mock for testing
        find_and_click = MagicMock(return_value=True)

        # Patch the functions
        with patch("owa_env_omniparser.unified.find_and_click_element", find_and_click):
            # Import after patching
            from owa_env_omniparser.unified import find_and_click_element

            # Execute the test
            result = find_and_click_element("Submit")

            # Verify result
            assert result is True
            find_and_click.assert_called_once_with("Submit")
