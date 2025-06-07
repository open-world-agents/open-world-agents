#!/usr/bin/env python3
"""
Demonstration of OEP-0003: Entry Points-Based Plugin Discovery and Unified Component Naming

This script demonstrates the new plugin system features:
1. Automatic plugin discovery via entry points
2. Unified namespace/name component naming
3. Enhanced component access API
4. Backwards compatibility with legacy activation
"""

import time

from owa.core.registry import (
    CALLABLES,
    LISTENERS,
    RUNNABLES,
    entry_point_registry,
    get_component,
    list_components,
)


def main():
    print("=== OEP-0003 Plugin System Demonstration ===\n")

    # 1. Show automatic plugin discovery
    print("1. Automatic Plugin Discovery via Entry Points:")
    print(f"   Entry point registry initialized: {entry_point_registry is not None}")
    print(f"   Discovered plugins: {list(entry_point_registry.discovered_plugins.keys())}")

    for name, plugin in entry_point_registry.discovered_plugins.items():
        print(f"   - {name} v{plugin.version}: {plugin.description}")
    print()

    # 2. Show unified naming system
    print("2. Unified Component Naming (namespace/name):")
    print("   Unified naming:")
    print(f"   - std/time_ns: {CALLABLES.get('std/time_ns')}")
    print(f"   - std/tick: {LISTENERS.get('std/tick')}")
    print()

    # 3. Test component functionality
    print("3. Component Functionality:")

    # Test unified naming
    time_func = CALLABLES["std/time_ns"]
    current_time = time_func()
    print(f"   Current time (std/time_ns): {current_time}")
    print()

    # 4. Enhanced component access API
    print("4. Enhanced Component Access API:")

    # Get specific component
    time_func_api = get_component("callables", namespace="std", name="time_ns")
    print(f"   get_component('callables', namespace='std', name='time_ns'): {time_func_api}")

    # Get all components in namespace
    std_callables = get_component("callables", namespace="std")
    print(f"   get_component('callables', namespace='std'): {list(std_callables.keys())}")

    # List all components
    all_components = list_components()
    print(f"   list_components() component types: {list(all_components.keys())}")

    # List components by namespace
    std_components = list_components(namespace="std")
    print(f"   list_components(namespace='std'):")
    for comp_type, components in std_components.items():
        if components:
            print(f"     {comp_type}: {components}")
    print()

    # 5. Entry Points Only System
    print("5. Entry Points Only System:")
    print("   Only entry points-based plugins are supported")
    print("   Manual activation has been removed")
    print("   All plugins must use entry points for discovery")
    print()

    # 6. Show tick listener in action
    print("6. Tick Listener Demonstration:")
    print("   Starting std/tick listener for 3 seconds...")

    tick_count = 0

    def tick_callback():
        nonlocal tick_count
        tick_count += 1
        current = CALLABLES["std/time_ns"]()
        print(f"   Tick #{tick_count} at {current}")

    # Use new unified naming
    tick_listener = LISTENERS["std/tick"]()
    configured_listener = tick_listener.configure(callback=tick_callback, interval=1)

    configured_listener.start()
    time.sleep(3.2)  # Let it tick a few times
    configured_listener.stop()
    configured_listener.join()

    print(f"   Received {tick_count} ticks")
    print()

    print("=== OEP-0003 Demonstration Complete ===")
    print("\nKey Benefits:")
    print("✓ Zero-configuration plugin discovery")
    print("✓ Unified namespace/name component naming")
    print("✓ Enhanced component access API")
    print("✓ Clean, modern plugin architecture")
    print("✓ Standard Python packaging integration")


if __name__ == "__main__":
    main()
