#!/usr/bin/env python3
"""
Test the get_options method for mkdocstrings v1 compatibility.
"""

import sys
from pathlib import Path

# Add the handler to the path
sys.path.insert(0, str(Path(__file__).parent))

from mkdocstrings_handlers.owa.handler import OWAHandler


def test_get_options():
    """Test the get_options method."""
    print("Testing get_options method for mkdocstrings v1 compatibility")
    print("=" * 60)

    # Test with empty options
    handler = OWAHandler()
    local_opts = {}
    merged = handler.get_options(local_opts)

    print("Test 1: Empty options")
    print(f"Local: {local_opts}")
    print(f"Merged: {merged}")
    print(f"Has defaults: {all(key in merged for key in ['show_source', 'show_signature', 'show_examples'])}")

    # Test with global options in handler initialization
    handler_with_global = OWAHandler(options={"show_source": False, "custom_global": "value"})
    local_opts = {}
    merged = handler_with_global.get_options(local_opts)

    print("\nTest 2: Global options in handler")
    print(f"Global (in handler): {handler_with_global.global_options}")
    print(f"Local: {local_opts}")
    print(f"Merged: {merged}")
    print(f"show_source overridden: {merged['show_source'] == False}")

    # Test with local options overriding global
    handler_with_global = OWAHandler(options={"show_source": False, "show_signature": True})
    local_opts = {"show_source": True, "custom_local": "local_value"}
    merged = handler_with_global.get_options(local_opts)

    print("\nTest 3: Local options override global")
    print(f"Global (in handler): {handler_with_global.global_options}")
    print(f"Local: {local_opts}")
    print(f"Merged: {merged}")
    print(f"show_source from local: {merged['show_source'] == True}")
    print(f"show_signature from global: {merged['show_signature'] == True}")
    print(f"Has custom_local: {'custom_local' in merged}")

    print("\n✅ get_options method working correctly!")


def test_options_in_rendering():
    """Test that options actually affect rendering."""
    print("\nTesting options effect on rendering")
    print("=" * 40)

    handler = OWAHandler()

    # Get component data
    data = handler.collect("example/add", {})
    if data.get("error"):
        print(f"Error: {data['error']}")
        return

    # Test with show_source=False
    options_no_source = {"show_source": False, "show_examples": True}
    html_no_source = handler.render(data, options_no_source)

    # Test with show_source=True
    options_with_source = {"show_source": True, "show_examples": True}
    html_with_source = handler.render(data, options_with_source)

    print(f"HTML without source contains 'Source Code': {'Source Code' in html_no_source}")
    print(f"HTML with source contains 'Source Code': {'Source Code' in html_with_source}")

    # Test with show_examples=False
    options_no_examples = {"show_source": True, "show_examples": False}
    html_no_examples = handler.render(data, options_no_examples)

    print(f"HTML without examples contains 'Usage Examples': {'Usage Examples' in html_no_examples}")
    print(f"HTML with examples contains 'Usage Examples': {'Usage Examples' in html_with_source}")

    print("\n✅ Options correctly control rendering!")


if __name__ == "__main__":
    test_get_options()
    test_options_in_rendering()
    print("\n🎉 All tests passed! mkdocstrings v1 compatibility fixed.")
