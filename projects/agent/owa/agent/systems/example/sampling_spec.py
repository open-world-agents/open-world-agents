from loguru import logger

from owa.agent.core.spec import ContinuousSamplingStrategy, DiscreteSamplingStrategy, PerceptionSamplingSpec
from owa.env.desktop.msg import KeyboardEvent, KeyboardState, MouseEvent, MouseState


def update_keyboard_state(state: KeyboardState, new_event: KeyboardEvent):
    if new_event.event_type == "press":
        state.buttons.add(new_event.vk)
    elif new_event.event_type == "release":
        try:
            state.buttons.remove(new_event.vk)
        except KeyError:
            logger.warning(f"Key release event processed without a prior press: {new_event}")
    return state


def keyboard_state_to_event(state: KeyboardState):
    """Convert keyboard state to a list of events."""
    events = []
    # Eliminates the mouse key events (left, right, middle, x1, x2)
    for key in state.buttons - {1, 2, 4, 5, 6}:
        events.append(KeyboardEvent(event_type="press", vk=key))
    return events


def update_mouse_state(state: MouseState, new_event: MouseEvent):
    if new_event.event_type == "move":
        state.x = new_event.x
        state.y = new_event.y
    elif new_event.event_type == "click":
        if new_event.pressed:
            state.buttons.add(new_event.button)
        else:
            try:
                state.buttons.remove(new_event.button)
            except KeyError:
                logger.warning(f"Mouse button release event processed without a prior press: {new_event}")
    elif new_event.event_type == "scroll":
        # Handle scroll events if needed
        pass
    return state


def mouse_state_to_event(state: MouseState):
    """Convert mouse state to a list of events."""
    events = []
    events.append(MouseEvent(event_type="move", x=state.x, y=state.y))
    # Note that following code will generate events which are NOT equivalent to real events:
    #   mouse position (x, y) where the click happens is unknown, so we use the last known position, but it's just a GUESS.
    for button in state.buttons:
        events.append(MouseEvent(event_type="click", x=state.x, y=state.y, button=button, pressed=True))
    return events


PERCEPTION_SAMPLING_SPEC = PerceptionSamplingSpec(
    **{
        # Screen - continuous event with 20 fps
        "inputs/screen": ContinuousSamplingStrategy(
            topic="screen",
            window_start=-0.25 - 0.05,  # 0.05 seconds margin to ensure k=5
            window_end=0,
            mode="last_k",  # Get the last k frames
            k=5,
            fps=20.0,  # sample from continuous events with 20fps = 0.05 seconds per frame
        ),
        # Mouse move - continuous event with 20 fps
        "inputs/mouse/move": ContinuousSamplingStrategy(
            topic="mouse",
            msg_filter=lambda x: x.event_type == "move",
            window_start=-0.5 - 0.05,  # 0.05 seconds margin to ensure k=5
            window_end=0,
            mode="last_k",
            k=5,
            fps=20.0,
        ),
        # Mouse click/scroll - discrete event, get all events in window
        "inputs/mouse/click-scroll": DiscreteSamplingStrategy(
            topic="mouse",
            msg_filter=lambda x: x.event_type in ("click", "scroll"),
            window_start=-0.5,
            window_end=0,
            mode="all",
            include_prior_state=True,
            state_topic="mouse/state",
            state_update_fn=update_mouse_state,
            state_to_event_fn=mouse_state_to_event,
        ),
        # Keyboard - discrete event, get all events in window
        "inputs/keyboard": DiscreteSamplingStrategy(
            topic="keyboard",
            window_start=-0.5,
            window_end=0,
            mode="all",
            include_prior_state=True,
            state_topic="keyboard/state",
            state_update_fn=update_keyboard_state,
            state_to_event_fn=keyboard_state_to_event,
        ),
        # Mouse move label - continuous event with 20 fps
        "outputs/mouse/move": ContinuousSamplingStrategy(
            topic="mouse",
            msg_filter=lambda x: x.event_type == "move",
            window_start=0,
            window_end=1,
            mode="all",
            fps=20.0,  # 1/0.05 = 20 fps
        ),
        # Mouse click/scroll label - discrete event, get all events in label window
        "outputs/mouse/click-scroll": DiscreteSamplingStrategy(
            topic="mouse",
            msg_filter=lambda x: x.event_type in ("click", "scroll"),
            window_start=0,
            window_end=1,
            mode="all",
        ),
        # Keyboard label - discrete event, get all events in label window
        "outputs/keyboard": DiscreteSamplingStrategy(
            topic="keyboard",
            window_start=0,
            window_end=1,
            mode="all",
        ),
    }
)
