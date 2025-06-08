#!/usr/bin/env python3
"""
Demonstration of the enhanced verbose output for OWA mkdocstrings handler.

This script shows the rich information now available for EnvPlugin documentation.
"""

import sys
from pathlib import Path

# Add the handler to the path
sys.path.insert(0, str(Path(__file__).parent))

from mkdocstrings_handlers.owa.handler import OWAHandler


def demo_component_analysis():
    """Demonstrate the comprehensive component analysis."""
    print("🎯 Enhanced Component Analysis Demo")
    print("=" * 50)
    
    handler = OWAHandler()
    
    # Analyze a component with Griffe
    component_data = handler.collect("example/add", {})
    
    if component_data.get("error"):
        print(f"❌ Error: {component_data['error']}")
        return
    
    print(f"📦 Component: {component_data['full_name']}")
    print(f"🏷️  Type: {component_data['component_type']}")
    print(f"📂 Import: {component_data['import_path']}")
    
    # Show signature information
    sig_info = component_data.get("signature_info", {})
    if sig_info:
        print(f"\n📝 Signature: {sig_info.get('signature_str', 'N/A')}")
        
        # Show Griffe parameters if available
        griffe_params = sig_info.get("griffe_parameters", [])
        if griffe_params:
            print("📋 Parameters (from Griffe):")
            for param in griffe_params:
                print(f"  • {param['name']}: {param.get('annotation', 'Any')}")
        
        # Show Griffe metadata
        griffe_info = sig_info.get("griffe_info", {})
        if griffe_info:
            print(f"🔍 Source Location: Line {griffe_info.get('lineno', 'Unknown')}")
            print(f"📊 Component Kind: {griffe_info.get('kind', 'Unknown')}")
    
    # Show parsed docstring
    parsed_doc = component_data.get("parsed_docstring", {})
    if parsed_doc:
        print(f"\n📚 Documentation:")
        if parsed_doc.get("summary"):
            print(f"  Summary: {parsed_doc['summary']}")
        if parsed_doc.get("args"):
            print(f"  Arguments: {len(parsed_doc['args'])} documented")
        if parsed_doc.get("returns"):
            print(f"  Returns: {parsed_doc['returns'][:50]}...")
    
    # Show usage examples
    examples = component_data.get("usage_examples", [])
    if examples:
        print(f"\n💡 Usage Examples ({len(examples)}):")
        for i, example in enumerate(examples[:2], 1):  # Show first 2
            print(f"  {i}. {example}")
    
    # Show class information
    class_info = component_data.get("component_class", {})
    if class_info:
        print(f"\n🔧 Component Type Info:")
        print(f"  • Is Function: {class_info.get('is_function', False)}")
        print(f"  • Is Class: {class_info.get('is_class', False)}")
        print(f"  • Module: {class_info.get('module', 'Unknown')}")


def demo_plugin_overview():
    """Demonstrate the enhanced plugin overview."""
    print("\n\n🔌 Enhanced Plugin Overview Demo")
    print("=" * 50)
    
    handler = OWAHandler()
    
    # Get plugin data
    plugin_data = handler.collect("example", {})
    
    if plugin_data.get("error"):
        print(f"❌ Error: {plugin_data['error']}")
        return
    
    print(f"📦 Plugin: {plugin_data['namespace']}")
    print(f"🏷️  Version: {plugin_data['version']}")
    print(f"📝 Description: {plugin_data['description']}")
    print(f"👤 Author: {plugin_data.get('author', 'Unknown')}")
    
    # Component statistics
    components = plugin_data.get("components", {})
    total = sum(len(comps) for comps in components.values())
    print(f"\n📊 Total Components: {total}")
    
    for comp_type, comp_list in components.items():
        if comp_list:
            print(f"  • {comp_type.title()}: {len(comp_list)}")
            
            # Show first component details
            first_comp = comp_list[0]
            if not first_comp.get("error"):
                summary = ""
                if first_comp.get("parsed_docstring", {}).get("summary"):
                    summary = first_comp["parsed_docstring"]["summary"][:60] + "..."
                print(f"    └─ {first_comp['name']}: {summary}")


def demo_griffe_vs_inspect():
    """Compare Griffe vs inspect-based analysis."""
    print("\n\n🔍 Griffe vs Inspect Comparison")
    print("=" * 50)
    
    handler = OWAHandler()
    
    # Get component data (which includes both)
    data = handler.collect("example/add", {})
    
    if data.get("error"):
        print(f"❌ Error: {data['error']}")
        return
    
    sig_info = data.get("signature_info", {})
    griffe_data = data.get("griffe_data", {})
    
    print("📊 Information Sources:")
    print(f"  • Griffe Available: {bool(griffe_data and not griffe_data.get('error'))}")
    print(f"  • Inspect Fallback: {bool(sig_info.get('signature_str'))}")
    
    if griffe_data and not griffe_data.get("error"):
        print(f"\n🎯 Griffe Analysis:")
        print(f"  • Kind: {griffe_data.get('kind')}")
        print(f"  • Parameters: {len(griffe_data.get('parameters', []))}")
        print(f"  • Line Range: {griffe_data.get('lineno')}-{griffe_data.get('endlineno')}")
        print(f"  • Has Docstring: {bool(griffe_data.get('docstring_raw'))}")
        
        # Show parameter details
        params = griffe_data.get("parameters", [])
        if params:
            print("  • Parameter Details:")
            for param in params:
                print(f"    - {param['name']}: {param.get('annotation', 'Any')} "
                      f"(default: {param.get('default', 'None')})")


def main():
    """Run the demonstration."""
    print("🚀 OWA mkdocstrings Handler - Verbose Output Demo")
    print("=" * 60)
    print("This demo shows the enhanced information now available")
    print("for EnvPlugin documentation rendering.")
    print("=" * 60)
    
    try:
        demo_component_analysis()
        demo_plugin_overview()
        demo_griffe_vs_inspect()
        
        print("\n\n✅ Demo completed successfully!")
        print("The handler now provides much more comprehensive")
        print("information for EnvPlugin users including:")
        print("  • Detailed parameter information with types")
        print("  • Enhanced docstring parsing")
        print("  • Source code locations")
        print("  • Usage examples")
        print("  • Component statistics")
        print("  • Griffe-powered analysis")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
