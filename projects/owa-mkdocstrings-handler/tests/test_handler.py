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


def test_handler_without_owa_core():
    """Test that the handler works even when owa-core is not available."""
    # This test verifies the import fallback mechanism
    import sys

    # Temporarily hide owa.core modules
    original_modules = {}
    owa_modules = [name for name in sys.modules.keys() if name.startswith("owa.core")]

    for module_name in owa_modules:
        original_modules[module_name] = sys.modules.pop(module_name, None)

    try:
        # Force reimport to test fallback
        import importlib

        if "mkdocstrings_handlers.owa.handler" in sys.modules:
            importlib.reload(sys.modules["mkdocstrings_handlers.owa.handler"])

        from mkdocstrings_handlers.owa import get_handler

        handler = get_handler()

        # Should still work with dummy implementations
        assert handler.name == "owa"

        # Should return error for any collection attempt
        result = handler.collect("test", {})
        assert "error" in result

    finally:
        # Restore original modules
        for module_name, module in original_modules.items():
            if module is not None:
                sys.modules[module_name] = module


if __name__ == "__main__":
    test_handler_import()
    test_handler_graceful_fallback()
    print("All tests passed!")
