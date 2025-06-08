#!/usr/bin/env python3
"""
Test script for the enhanced OWA mkdocstrings handler.

This script demonstrates the new verbose output capabilities with Griffe integration.
"""

import sys
from pathlib import Path

# Add the handler to the path
sys.path.insert(0, str(Path(__file__).parent))

from mkdocstrings_handlers.owa.handler import OWAHandler


def test_plugin_rendering():
    """Test plugin-level rendering with enhanced information."""
    print("=" * 60)
    print("Testing Plugin-level Rendering")
    print("=" * 60)
    
    handler = OWAHandler()
    
    # Test with example plugin
    try:
        data = handler.collect("example", {})
        if data.get("error"):
            print(f"Error collecting example plugin: {data['error']}")
            return
        
        html = handler.render(data, {})
        print("Plugin HTML (first 1000 chars):")
        print(html[:1000])
        print("..." if len(html) > 1000 else "")
        
    except Exception as e:
        print(f"Error testing plugin rendering: {e}")


def test_component_rendering():
    """Test component-level rendering with enhanced information."""
    print("\n" + "=" * 60)
    print("Testing Component-level Rendering")
    print("=" * 60)
    
    handler = OWAHandler()
    
    # Test with example component
    try:
        data = handler.collect("example/add", {})
        if data.get("error"):
            print(f"Error collecting example/add component: {data['error']}")
            return
        
        html = handler.render(data, {})
        print("Component HTML (first 1500 chars):")
        print(html[:1500])
        print("..." if len(html) > 1500 else "")
        
    except Exception as e:
        print(f"Error testing component rendering: {e}")


def test_griffe_integration():
    """Test Griffe integration specifically."""
    print("\n" + "=" * 60)
    print("Testing Griffe Integration")
    print("=" * 60)
    
    handler = OWAHandler()
    
    # Test Griffe analysis directly
    try:
        griffe_data = handler._analyze_with_griffe("owa.env.example.example_callable:example_add")
        if griffe_data:
            print("Griffe analysis successful!")
            print(f"Kind: {griffe_data.get('kind')}")
            print(f"Parameters: {len(griffe_data.get('parameters', []))}")
            print(f"Has docstring: {bool(griffe_data.get('docstring_raw'))}")
            print(f"Returns: {griffe_data.get('returns')}")
            
            if griffe_data.get("error"):
                print(f"Griffe error: {griffe_data['error']}")
        else:
            print("Griffe analysis returned None")
            
    except Exception as e:
        print(f"Error testing Griffe integration: {e}")


def main():
    """Run all tests."""
    print("Enhanced OWA mkdocstrings Handler Test")
    print("=====================================")
    
    # Check if dependencies are available
    try:
        from mkdocstrings_handlers.owa.handler import MKDOCSTRINGS_AVAILABLE, GRIFFE_AVAILABLE
        print(f"mkdocstrings available: {MKDOCSTRINGS_AVAILABLE}")
        print(f"griffe available: {GRIFFE_AVAILABLE}")
    except ImportError as e:
        print(f"Import error: {e}")
        return
    
    test_griffe_integration()
    test_plugin_rendering()
    test_component_rendering()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
