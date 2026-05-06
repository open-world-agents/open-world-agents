"""
Desktop gamepad message definitions.

This module contains message types for gamepad events and state,
following the domain-based message naming convention for better organization.
"""

from typing import Literal, TypeAlias

from owa.core.message import OWAMessage

# Matches definition of SDL3
# https://wiki.libsdl.org/SDL3/SDL_GamepadType
GamepadType: TypeAlias = Literal[
    "GAMEPAD_TYPE_STANDARD",
    "GAMEPAD_TYPE_XBOX360",
    "GAMEPAD_TYPE_XBOXONE",
    "GAMEPAD_TYPE_PS3",
    "GAMEPAD_TYPE_PS4",
    "GAMEPAD_TYPE_PS5",
    "GAMEPAD_TYPE_NINTENDO_SWITCH_PRO",
    "GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT",
    "GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT",
    "GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR",
    "GAMEPAD_TYPE_GAMECUBE",
    "GAMEPAD_TYPE_COUNT",
]


# https://wiki.libsdl.org/SDL3/SDL_GamepadButton
GamepadButton: TypeAlias = Literal[
    "GAMEPAD_BUTTON_INVALID",  # Invalid button (no button)
    "GAMEPAD_BUTTON_SOUTH",  # Bottom face button (e.g. Xbox A or PlayStation Cross)
    "GAMEPAD_BUTTON_EAST",  # Right face button (e.g. Xbox B or PlayStation Circle)
    "GAMEPAD_BUTTON_WEST",  # Left face button (e.g. Xbox X or PlayStation Square)
    "GAMEPAD_BUTTON_NORTH",  # Top face button (e.g. Xbox Y or PlayStation Triangle)
    "GAMEPAD_BUTTON_BACK",  # Back/select button (secondary menu)
    "GAMEPAD_BUTTON_GUIDE",  # Guide/PS/Home button
    "GAMEPAD_BUTTON_START",  # Start button (primary menu)
    "GAMEPAD_BUTTON_LEFT_STICK",  # Pressing the left analog stick
    "GAMEPAD_BUTTON_RIGHT_STICK",  # Pressing the right analog stick
    "GAMEPAD_BUTTON_LEFT_SHOULDER",  # Left shoulder button (LB/L1)
    "GAMEPAD_BUTTON_RIGHT_SHOULDER",  # Right shoulder button (RB/R1)
    "GAMEPAD_BUTTON_DPAD_UP",  # D-pad up
    "GAMEPAD_BUTTON_DPAD_DOWN",  # D-pad down
    "GAMEPAD_BUTTON_DPAD_LEFT",  # D-pad left
    "GAMEPAD_BUTTON_DPAD_RIGHT",  # D-pad right
    "GAMEPAD_BUTTON_MISC1",  # Additional misc button (e.g. Xbox share)
    "GAMEPAD_BUTTON_RIGHT_PADDLE1",  # First right paddle (e.g. Elite P1)
    "GAMEPAD_BUTTON_LEFT_PADDLE1",  # First left paddle (e.g. Elite P3)
    "GAMEPAD_BUTTON_RIGHT_PADDLE2",  # Second right paddle (e.g. Elite P2)
    "GAMEPAD_BUTTON_LEFT_PADDLE2",  # Second left paddle (e.g. Elite P4)
    "GAMEPAD_BUTTON_TOUCHPAD",  # Touchpad button (PS4/PS5)
    "GAMEPAD_BUTTON_MISC2",  # Additional misc button 2
    "GAMEPAD_BUTTON_MISC3",  # Additional misc button 3
    "GAMEPAD_BUTTON_MISC4",  # Additional misc button 4
    "GAMEPAD_BUTTON_MISC5",  # Additional misc button 5
    "GAMEPAD_BUTTON_MISC6",  # Additional misc button 6
    "GAMEPAD_BUTTON_COUNT",  # Count of buttons (not a button)
]


# https://wiki.libsdl.org/SDL3/SDL_GamepadAxis
GamepadAxis: TypeAlias = Literal[
    "GAMEPAD_AXIS_INVALID",  # Invalid axis (no axis)
    "GAMEPAD_AXIS_LEFT_X",  # Left stick X-axis
    "GAMEPAD_AXIS_LEFT_Y",  # Left stick Y-axis
    "GAMEPAD_AXIS_RIGHT_X",  # Right stick X-axis
    "GAMEPAD_AXIS_RIGHT_Y",  # Right stick Y-axis
    "GAMEPAD_AXIS_LEFT_TRIGGER",  # Left trigger axis
    "GAMEPAD_AXIS_RIGHT_TRIGGER",  # Right trigger axis
    "GAMEPAD_AXIS_COUNT",  # Count of axes (not an axis)
]


class GamepadEvent(OWAMessage):
    """
    Represents a gamepad event (button press, button release, axis motion).

    This message captures gamepad interactions with detailed event information,
    suitable for recording user interactions and replaying them.

    Attributes:
        event_type: Type of event - "button" or "axis"
        gamepad_type: Type/model of the gamepad (e.g., Xbox, PlayStation, etc.)
        button: Gamepad button involved (for button events)
        axis: Gamepad axis involved (for axis events)
        pressed: Whether button was pressed (True) or released (False) for button events
        value: Axis value (typically -1.0 to 1.0 for sticks, 0.0 to 1.0 for triggers)
        timestamp: Optional timestamp in nanoseconds since epoch
    """

    _type = "desktop/GamepadEvent"

    event_type: Literal["button", "axis"]
    gamepad_type: GamepadType
    button: GamepadButton | None = None
    axis: GamepadAxis | None = None
    pressed: bool | None = None
    value: float | None = None
    timestamp: int | None = None


class GamepadState(OWAMessage):
    """
    Represents the current state of a gamepad.

    This message captures the complete gamepad state at a point in time,
    useful for state synchronization and debugging.

    Attributes:
        gamepad_type: Type/model of the gamepad (e.g., Xbox, PlayStation, etc.)
        buttons: Set of currently pressed gamepad buttons
        axes: Dictionary mapping axis names to their current values
        timestamp: Optional timestamp in nanoseconds since epoch
    """

    _type = "desktop/GamepadState"

    gamepad_type: GamepadType
    buttons: set[GamepadButton]
    axes: dict[GamepadAxis, float]
    timestamp: int | None = None
