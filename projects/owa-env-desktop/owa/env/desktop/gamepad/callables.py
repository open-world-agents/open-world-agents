"""
Gamepad callable functions for retrieving gamepad state and information.

This module provides functions to get the current state of connected gamepads,
similar to how keyboard and mouse callables work.
"""

import time
from typing import List, Optional

try:
    import inputs
    INPUTS_AVAILABLE = True
except ImportError:
    INPUTS_AVAILABLE = False

from owa.msgs.desktop.gamepad import GamepadEvent, GamepadState, GamepadType, GamepadButton, GamepadAxis


def _detect_gamepad_type(device) -> GamepadType:
    """
    Detect the type of gamepad based on device information.
    
    Args:
        device: The gamepad device from inputs library
        
    Returns:
        GamepadType: The detected gamepad type
    """
    device_name = getattr(device, 'name', '').lower()
    
    if 'xbox 360' in device_name:
        return "GAMEPAD_TYPE_XBOX360"
    elif 'xbox one' in device_name or 'xbox wireless' in device_name:
        return "GAMEPAD_TYPE_XBOXONE"
    elif 'ps3' in device_name or 'playstation 3' in device_name:
        return "GAMEPAD_TYPE_PS3"
    elif 'ps4' in device_name or 'playstation 4' in device_name:
        return "GAMEPAD_TYPE_PS4"
    elif 'ps5' in device_name or 'playstation 5' in device_name:
        return "GAMEPAD_TYPE_PS5"
    elif 'nintendo' in device_name:
        if 'pro' in device_name:
            return "GAMEPAD_TYPE_NINTENDO_SWITCH_PRO"
        elif 'joy-con' in device_name:
            if 'left' in device_name:
                return "GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT"
            elif 'right' in device_name:
                return "GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT"
            else:
                return "GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR"
        else:
            return "GAMEPAD_TYPE_NINTENDO_SWITCH_PRO"
    elif 'gamecube' in device_name:
        return "GAMEPAD_TYPE_GAMECUBE"
    else:
        return "GAMEPAD_TYPE_STANDARD"


def _map_button_code(code: str) -> GamepadButton:
    """Map inputs library button codes to GamepadButton enum values."""
    button_map = {
        'BTN_A': "GAMEPAD_BUTTON_SOUTH",
        'BTN_B': "GAMEPAD_BUTTON_EAST", 
        'BTN_X': "GAMEPAD_BUTTON_WEST",
        'BTN_Y': "GAMEPAD_BUTTON_NORTH",
        'BTN_LB': "GAMEPAD_BUTTON_LEFT_SHOULDER",
        'BTN_RB': "GAMEPAD_BUTTON_RIGHT_SHOULDER",
        'BTN_BACK': "GAMEPAD_BUTTON_BACK",
        'BTN_START': "GAMEPAD_BUTTON_START",
        'BTN_MODE': "GAMEPAD_BUTTON_GUIDE",
        'BTN_THUMBL': "GAMEPAD_BUTTON_LEFT_STICK",
        'BTN_THUMBR': "GAMEPAD_BUTTON_RIGHT_STICK",
        'BTN_DPAD_UP': "GAMEPAD_BUTTON_DPAD_UP",
        'BTN_DPAD_DOWN': "GAMEPAD_BUTTON_DPAD_DOWN",
        'BTN_DPAD_LEFT': "GAMEPAD_BUTTON_DPAD_LEFT",
        'BTN_DPAD_RIGHT': "GAMEPAD_BUTTON_DPAD_RIGHT",
    }
    return button_map.get(code, "GAMEPAD_BUTTON_INVALID")


def _map_axis_code(code: str) -> GamepadAxis:
    """Map inputs library axis codes to GamepadAxis enum values."""
    axis_map = {
        'ABS_X': "GAMEPAD_AXIS_LEFT_X",
        'ABS_Y': "GAMEPAD_AXIS_LEFT_Y", 
        'ABS_RX': "GAMEPAD_AXIS_RIGHT_X",
        'ABS_RY': "GAMEPAD_AXIS_RIGHT_Y",
        'ABS_Z': "GAMEPAD_AXIS_LEFT_TRIGGER",
        'ABS_RZ': "GAMEPAD_AXIS_RIGHT_TRIGGER",
    }
    return axis_map.get(code, "GAMEPAD_AXIS_INVALID")


def get_gamepad_state(gamepad_index: int = 0) -> Optional[GamepadState]:
    """
    Get the current state of a specific gamepad.

    Args:
        gamepad_index: Index of the gamepad to query (0 for first gamepad)

    Returns:
        GamepadState: Current state of the gamepad, or None if not available

    Examples:
        >>> state = get_gamepad_state(0)  # Get state of first gamepad
        >>> if state and state.buttons:
        ...     print(f"Pressed buttons: {state.buttons}")
        >>> if state and state.axes:
        ...     print(f"Left stick X: {state.axes.get('GAMEPAD_AXIS_LEFT_X', 0)}")

    Raises:
        ImportError: If the 'inputs' library is not installed
    """
    if not INPUTS_AVAILABLE:
        raise ImportError(
            "The 'inputs' library is required for gamepad support. "
            "Install it with: pip install inputs"
        )

    try:
        # Get all gamepad devices
        devices = inputs.DeviceManager()
        gamepads = list(devices.gamepads)
        
        if gamepad_index >= len(gamepads):
            return None
            
        device = gamepads[gamepad_index]
        gamepad_type = _detect_gamepad_type(device)
        
        # Initialize state tracking
        buttons = set()
        axes = {}
        
        # Read current state by processing all pending events
        try:
            events = inputs.get_gamepad()
            for event in events:
                if hasattr(event, 'device') and event.device == device:
                    # Process button events
                    if event.ev_type == 'Key':
                        button = _map_button_code(event.code)
                        if button != "GAMEPAD_BUTTON_INVALID":
                            if event.state:
                                buttons.add(button)
                            else:
                                buttons.discard(button)
                    
                    # Process axis events
                    elif event.ev_type == 'Absolute':
                        axis = _map_axis_code(event.code)
                        if axis != "GAMEPAD_AXIS_INVALID":
                            # Normalize axis values
                            if axis in ["GAMEPAD_AXIS_LEFT_TRIGGER", "GAMEPAD_AXIS_RIGHT_TRIGGER"]:
                                normalized_value = max(0.0, min(1.0, event.state / 255.0))
                            else:
                                normalized_value = max(-1.0, min(1.0, event.state / 32767.0))
                            
                            axes[axis] = normalized_value
        except inputs.UnpluggedError:
            return None
        except Exception:
            # Return empty state if we can't read events
            pass
        
        return GamepadState(
            gamepad_type=gamepad_type,
            buttons=buttons,
            axes=axes,
            timestamp=time.time_ns()
        )
        
    except Exception:
        return None


def get_connected_gamepads() -> List[dict]:
    """
    Get information about all connected gamepads.

    Returns:
        List[dict]: List of dictionaries containing gamepad information.
                   Each dict has 'index', 'name', and 'type' keys.

    Examples:
        >>> gamepads = get_connected_gamepads()
        >>> for gamepad in gamepads:
        ...     print(f"Gamepad {gamepad['index']}: {gamepad['name']} ({gamepad['type']})")

    Raises:
        ImportError: If the 'inputs' library is not installed
    """
    if not INPUTS_AVAILABLE:
        raise ImportError(
            "The 'inputs' library is required for gamepad support. "
            "Install it with: pip install inputs"
        )

    try:
        devices = inputs.DeviceManager()
        gamepads = []
        
        for i, device in enumerate(devices.gamepads):
            gamepads.append({
                'index': i,
                'name': getattr(device, 'name', f'Gamepad {i}'),
                'type': _detect_gamepad_type(device)
            })
        
        return gamepads
        
    except Exception:
        return []


def is_gamepad_connected(gamepad_index: int = 0) -> bool:
    """
    Check if a specific gamepad is connected.

    Args:
        gamepad_index: Index of the gamepad to check (0 for first gamepad)

    Returns:
        bool: True if the gamepad is connected, False otherwise

    Examples:
        >>> if is_gamepad_connected(0):
        ...     print("First gamepad is connected")
        >>> if is_gamepad_connected(1):
        ...     print("Second gamepad is connected")

    Raises:
        ImportError: If the 'inputs' library is not installed
    """
    if not INPUTS_AVAILABLE:
        raise ImportError(
            "The 'inputs' library is required for gamepad support. "
            "Install it with: pip install inputs"
        )

    try:
        devices = inputs.DeviceManager()
        gamepads = list(devices.gamepads)
        return gamepad_index < len(gamepads)
    except Exception:
        return False


def get_gamepad_events():
    """
    Get pending gamepad events from all connected gamepads.

    This function is used internally by gamepad listeners to get raw events
    from the inputs library and convert them to GamepadEvent messages.

    Returns:
        List of GamepadEvent objects for all pending events

    Raises:
        ImportError: If the 'inputs' library is not installed
    """
    if not INPUTS_AVAILABLE:
        raise ImportError(
            "The 'inputs' library is required for gamepad support. "
            "Install it with: pip install inputs"
        )

    events = []

    try:
        # Get all gamepad devices for type detection
        devices = inputs.DeviceManager()
        device_types = {}
        for device in devices.gamepads:
            device_types[device] = _detect_gamepad_type(device)

        # Get raw events from inputs library
        raw_events = inputs.get_gamepad()

        for event in raw_events:
            # Determine gamepad type for this event
            gamepad_type = "GAMEPAD_TYPE_STANDARD"  # Default
            if hasattr(event, 'device') and event.device in device_types:
                gamepad_type = device_types[event.device]

            # Process button events
            if event.ev_type == 'Key':
                button = _map_button_code(event.code)
                if button != "GAMEPAD_BUTTON_INVALID":
                    gamepad_event = GamepadEvent(
                        event_type="button",
                        gamepad_type=gamepad_type,
                        button=button,
                        pressed=bool(event.state),
                        timestamp=time.time_ns()
                    )
                    events.append(gamepad_event)

            # Process axis events
            elif event.ev_type == 'Absolute':
                axis = _map_axis_code(event.code)
                if axis != "GAMEPAD_AXIS_INVALID":
                    # Normalize axis values to -1.0 to 1.0 range for sticks
                    # and 0.0 to 1.0 for triggers
                    if axis in ["GAMEPAD_AXIS_LEFT_TRIGGER", "GAMEPAD_AXIS_RIGHT_TRIGGER"]:
                        # Triggers: normalize from 0-255 to 0.0-1.0
                        normalized_value = max(0.0, min(1.0, event.state / 255.0))
                    else:
                        # Sticks: normalize from -32768-32767 to -1.0-1.0
                        normalized_value = max(-1.0, min(1.0, event.state / 32767.0))

                    gamepad_event = GamepadEvent(
                        event_type="axis",
                        gamepad_type=gamepad_type,
                        axis=axis,
                        value=normalized_value,
                        timestamp=time.time_ns()
                    )
                    events.append(gamepad_event)

    except inputs.UnpluggedError:
        # Gamepad was unplugged, return empty list
        pass
    except Exception:
        # Handle other exceptions gracefully
        pass

    return events
