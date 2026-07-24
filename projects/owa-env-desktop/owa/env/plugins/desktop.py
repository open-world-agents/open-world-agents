"""
Plugin specification for the Desktop environment plugin.

This module is kept separate to avoid circular imports during plugin discovery.
"""

from owa.core.plugin_spec import PluginSpec


def _get_package_version() -> str:
    """Get the version of the owa-env-desktop package."""
    try:
        from importlib.metadata import version
    except ImportError:  # For Python <3.8
        from importlib_metadata import version

    try:
        return version("owa-env-desktop")
    except Exception:
        return "unknown"


# Plugin specification for entry points discovery
plugin_spec = PluginSpec(
    namespace="desktop",
    version=_get_package_version(),
    description="Desktop environment plugin with mouse, keyboard, and window control",
    author="OWA Development Team",
    components={
        "callables": {
            # Screen capture
            "screen.capture": "owa.env.desktop.screen.callables:capture_screen",
            # Mouse control
            "mouse.click": "owa.env.desktop.keyboard_mouse.callables:click",
            "mouse.move": "owa.env.desktop.keyboard_mouse.callables:mouse_move",
            "mouse.position": "owa.env.desktop.keyboard_mouse.callables:mouse_position",
            "mouse.press": "owa.env.desktop.keyboard_mouse.callables:mouse_press",
            "mouse.release": "owa.env.desktop.keyboard_mouse.callables:mouse_release",
            "mouse.scroll": "owa.env.desktop.keyboard_mouse.callables:mouse_scroll",
            "mouse.get_state": "owa.env.desktop.keyboard_mouse.callables:get_mouse_state",
            "mouse.get_pointer_ballistics_config": "owa.env.desktop.keyboard_mouse.callables:get_pointer_ballistics_config",
            # Keyboard control
            "keyboard.press": "owa.env.desktop.keyboard_mouse.callables:press",
            "keyboard.release": "owa.env.desktop.keyboard_mouse.callables:release",
            "keyboard.type": "owa.env.desktop.keyboard_mouse.callables:keyboard_type",
            "keyboard.get_state": "owa.env.desktop.keyboard_mouse.callables:get_keyboard_state",
            "keyboard.press_repeat": "owa.env.desktop.keyboard_mouse.callables:press_repeat_key",
            "keyboard.release_all_keys": "owa.env.desktop.keyboard_mouse.callables:release_all_keys",
            "keyboard.get_keyboard_repeat_timing": "owa.env.desktop.keyboard_mouse.callables:get_keyboard_repeat_timing",
            # Window management
            "window.get_active_window": "owa.env.desktop.window.callables:get_active_window",
            "window.get_window_by_title": "owa.env.desktop.window.callables:get_window_by_title",
            "window.get_pid_by_title": "owa.env.desktop.window.callables:get_pid_by_title",
            "window.when_active": "owa.env.desktop.window.callables:when_active",
            "window.is_active": "owa.env.desktop.window.callables:is_active",
            "window.make_active": "owa.env.desktop.window.callables:make_active",
        },
        "listeners": {
            # Input listeners
            "keyboard": "owa.env.desktop.keyboard_mouse.listeners:KeyboardListenerWrapper",
            "mouse": "owa.env.desktop.keyboard_mouse.listeners:MouseListenerWrapper",
            "raw_mouse": "owa.env.desktop.keyboard_mouse.listeners:RawMouseListener",
            "keyboard_state": "owa.env.desktop.keyboard_mouse.listeners:KeyboardStateListener",
            "mouse_state": "owa.env.desktop.keyboard_mouse.listeners:MouseStateListener",
            # Window listener
            "window": "owa.env.desktop.window.listeners:WindowListener",
        },
    },
)
