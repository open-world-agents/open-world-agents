from loguru import logger

from owa.agent.core import Event
from owa.agent.core.perception import (
    PerceptionSpec,
    PerceptionSpecDict,
    SamplingConfig,
    TrimConfig,
)
from owa.env.desktop.msg import KeyboardEvent, KeyboardState, MouseEvent, MouseState


def update_keyboard_state(state: KeyboardState, new_event: KeyboardEvent):
    if new_event.event_type == "press":
        state.buttons.add(new_event.vk)
    elif new_event.event_type == "release":
        try:
            state.buttons.remove(new_event.vk)
        except KeyError:
            logger.warning(
                f"Key release event processed without a prior press: {new_event}"
            )
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
                logger.warning(
                    f"Mouse button release event processed without a prior press: {new_event}"
                )
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
        events.append(
            MouseEvent(
                event_type="click", x=state.x, y=state.y, button=button, pressed=True
            )
        )
    return events


# Functions for sampling and trimming decisions
def is_move_event(event: Event) -> bool:
    msg: MouseEvent = event.msg
    return msg.event_type == "move"


PERCEPTION_SPEC_DICT = PerceptionSpecDict(
    {
        # Screen - continuous event with 20 fps sampling
        "inputs/screen": PerceptionSpec(
            topics=["screen"],
            window_start=-0.50
            - 0.05,  # 0.05 seconds margin to ensure enough `k` samples
            window_end=0,
            sample_configs=[SamplingConfig(sampling_rate=10.0)],
            trim_configs=[TrimConfig(trim_mode="last_k", trim_k=5)],
        ),
        # Keyboard - discrete event, get all events in window
        "inputs/keyboard": PerceptionSpec(
            topics=["keyboard"], window_start=-0.5, window_end=0
        ),
        # Mouse - move(20Hz) and click/scroll(all)
        "inputs/mouse": PerceptionSpec(
            topics=["mouse"],
            window_start=-0.5,
            window_end=0,
            sample_configs=[
                SamplingConfig(sample_if=is_move_event, sampling_rate=20.0)
            ],
            trim_configs=[
                TrimConfig(trim_if=is_move_event, trim_mode="last_k", trim_k=5)
            ],
        ),
        # Keyboard/Mouse state - needed to ensure stateful processing of `click` or `press`/`release` events
        # FIXME: I've thought about this a lot, but I don't know the answer. Maybe some improvement is needed.
        "inputs/keyboard/state": PerceptionSpec(
            topics=["keyboard/state"],
            window_start=-0.5,
            window_end=0,
            trim_configs=[TrimConfig(trim_mode="first_k", trim_k=1)],
        ),
        "inputs/mouse/state": PerceptionSpec(
            topics=["mouse/state"],
            window_start=-0.5,
            window_end=0,
            trim_configs=[TrimConfig(trim_mode="first_k", trim_k=1)],
        ),
        # Keyboard label - discrete event, get all events in label window
        "outputs/keyboard": PerceptionSpec(
            topics=["keyboard"], window_start=0, window_end=1
        ),
        # Mouse move label - continuous event with 20 fps
        "outputs/mouse": PerceptionSpec(
            topics=["mouse"],
            window_start=0,
            window_end=1,
            sample_configs=[
                SamplingConfig(sample_if=is_move_event, sampling_rate=20.0)
            ],
        ),
    }
)


# PERCEPTION_SPEC_DICT = PerceptionSpecDict(
#     {
#         # Screen - continuous event with 20 fps sampling
#         "inputs/screen": PerceptionSpec(
#             topics=["screen"],
#             window_start=-0.50 - 0.05,  # 0.05 seconds margin to ensure enough `k` samples
#             window_end=0,
#             sample_configs=[SamplingConfig(sampling_rate=10.0)],
#             trim_configs=[TrimConfig(trim_mode="last_k", trim_k=5)],
#         ),
#         # Keyboard - discrete event, get all events in window
#         "inputs/keyboard": PerceptionSpec(topics=["keyboard"], window_start=-0.5, window_end=0),
#         # Keyboard/Mouse state - needed to ensure stateful processing of `click` or `press`/`release` events
#         # FIXME: I've thought about this a lot, but I don't know the answer. Maybe some improvement is needed.
#         "inputs/keyboard/state": PerceptionSpec(
#             topics=["keyboard/state"],
#             window_start=-0.5,
#             window_end=0,
#             trim_configs=[TrimConfig(trim_mode="first_k", trim_k=1)],
#         ),
#         # Keyboard label - discrete event, get all events in label window
#         "outputs/keyboard": PerceptionSpec(topics=["keyboard"], window_start=0, window_end=1),
#     }
# )
