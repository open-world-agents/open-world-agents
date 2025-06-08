"""
Tests for the OWA mkdocstrings handler.
"""


def test_handler_import():
    """Test that the handler can be imported successfully."""
    from mkdocstrings_handlers.owa import get_handler

    handler = get_handler()
    assert handler.name == "owa"
    assert handler.domain == "py"


def test_handler_graceful_fallback():
    """Test that the handler handles missing dependencies gracefully."""
    from mkdocstrings_handlers.owa import get_handler

    handler = get_handler()

    # Test with non-existent plugin
    result = handler.collect("nonexistent", {})
    assert "error" in result

    # Test rendering error cases
    error_data = {"error": "Test error"}
    rendered = handler.render(error_data, {})
    assert "error" in rendered.lower()


if __name__ == "__main__":
    test_handler_import()
    test_handler_graceful_fallback()
    print("All tests passed!")
