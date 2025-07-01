#!/usr/bin/env python3
"""
Example script demonstrating gamepad input collection using OWA gamepad listeners.

This script shows how to use the gamepad listeners to capture gamepad events
and state, similar to how keyboard and mouse listeners work.
"""

import time
from owa.core import CALLABLES, LISTENERS, MESSAGES

# Access message types through the global registry
GamepadEvent = MESSAGES["desktop/GamepadEvent"]
GamepadState = MESSAGES["desktop/GamepadState"]

print("🎮 Gamepad Input Example")
print("=" * 50)

# Check if any gamepads are connected
try:
    connected_gamepads = CALLABLES["gamepad.get_connected"]()
    if not connected_gamepads:
        print("❌ No gamepads detected. Please connect a gamepad and try again.")
        exit(1)
    
    print(f"✅ Found {len(connected_gamepads)} gamepad(s):")
    for gamepad in connected_gamepads:
        print(f"  - {gamepad['name']} (Type: {gamepad['type']})")
    print()

except ImportError as e:
    print(f"❌ Error: {e}")
    print("Please install the 'inputs' library: pip install inputs")
    exit(1)

# Example 1: Gamepad Event Listener
print("🎯 Starting gamepad event listener...")
print("Press any buttons or move sticks/triggers on your gamepad.")
print("Press Ctrl+C to stop.\n")

def on_gamepad_event(event: GamepadEvent):
    """Callback function for gamepad events."""
    if event.event_type == "button":
        action = "pressed" if event.pressed else "released"
        print(f"🔘 Button {event.button} {action} on {event.gamepad_type}")
    elif event.event_type == "axis":
        print(f"🕹️  Axis {event.axis}: {event.value:.3f} on {event.gamepad_type}")

def on_gamepad_state(state: GamepadState):
    """Callback function for gamepad state updates."""
    if state.buttons or any(abs(v) > 0.1 for v in state.axes.values()):
        print(f"📊 State - Buttons: {len(state.buttons)}, Active axes: {len([v for v in state.axes.values() if abs(v) > 0.1])}")

# Configure gamepad event listener
gamepad_listener = LISTENERS["gamepad"]().configure(callback=on_gamepad_event)

# Configure gamepad state listener (reports every second)
gamepad_state_listener = LISTENERS["gamepad_state"]().configure(callback=on_gamepad_state)

try:
    print("Starting listeners...")
    
    # Start both listeners
    gamepad_listener.start()
    gamepad_state_listener.start()
    
    print("✅ Listeners started! Try using your gamepad.")
    print("   - Button presses/releases will be shown immediately")
    print("   - Axis movements will be shown immediately") 
    print("   - Overall state summary will be shown every second")
    print("   - Press Ctrl+C to stop\n")
    
    # Keep the script running
    while True:
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("\n🛑 Stopping listeners...")
    
finally:
    # Clean up
    if 'gamepad_listener' in locals():
        gamepad_listener.stop()
        gamepad_listener.join()
    
    if 'gamepad_state_listener' in locals():
        gamepad_state_listener.stop()
        gamepad_state_listener.join()
    
    print("✅ Gamepad listeners stopped.")

print("\n🎮 Example completed!")
print("\nYou can now use these listeners in your own applications:")
print("```python")
print("from owa.core import LISTENERS, CALLABLES")
print("")
print("# Event-based listening")
print("def my_gamepad_callback(event):")
print("    print(f'Gamepad event: {event}')")
print("")
print("listener = LISTENERS['gamepad']().configure(callback=my_gamepad_callback)")
print("listener.start()")
print("# ... your application logic ...")
print("listener.stop()")
print("listener.join()")
print("")
print("# Or get gamepad state directly")
print("state = CALLABLES['gamepad.get_state'](0)  # Get state of first gamepad")
print("if state:")
print("    print(f'Buttons: {state.buttons}')")
print("    print(f'Axes: {state.axes}')")
print("```")
