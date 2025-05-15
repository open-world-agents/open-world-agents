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


PERCEPTION_SAMPLING_SPEC = PerceptionSamplingSpec(
    inputs=[
        # Screen - continuous event with 20 fps
        ContinuousSamplingStrategy(
            topic="screen",
            window_start=-0.25 - 0.05,  # 0.05 seconds margin to ensure k=5
            window_end=0,
            mode="last_k",  # Get the last k frames
            k=5,
            fps=20.0,  # sample from continuous events with 20fps = 0.05 seconds per frame
        ),
        # Mouse move - continuous event with 20 fps
        ContinuousSamplingStrategy(
            topic="mouse",
            msg_filter=lambda x: x.event_type == "move",
            window_start=-0.25 - 0.05,  # 0.05 seconds margin to ensure k=5
            window_end=0,
            mode="last_k",
            k=5,
            fps=20.0,
        ),
        # Mouse click/scroll - discrete event, get all events in window
        DiscreteSamplingStrategy(
            topic="mouse",
            msg_filter=lambda x: x.event_type in ("click", "scroll"),
            window_start=-0.25,
            window_end=0,
            mode="all",
            include_prior_state=True,
            state_topic="mouse/state",
            state_update_fn=update_mouse_state,
        ),
        # Keyboard - discrete event, get all events in window
        DiscreteSamplingStrategy(
            topic="keyboard",
            window_start=-0.25,
            window_end=0,
            mode="all",
            include_prior_state=True,
            state_topic="keyboard/state",
            state_update_fn=update_keyboard_state,
        ),
    ],
    outputs=[
        # Mouse move label - continuous event with 20 fps
        ContinuousSamplingStrategy(
            topic="mouse",
            msg_filter=lambda x: x.event_type == "move",
            window_start=0,
            window_end=1,
            mode="all",
            fps=20.0,  # 1/0.05 = 20 fps
        ),
        # Mouse click/scroll label - discrete event, get all events in label window
        DiscreteSamplingStrategy(
            topic="mouse",
            msg_filter=lambda x: x.event_type in ("click", "scroll"),
            window_start=0,
            window_end=1,
            mode="all",
        ),
        # Keyboard label - discrete event, get all events in label window
        DiscreteSamplingStrategy(
            topic="keyboard",
            window_start=0,
            window_end=1,
            mode="all",
        ),
    ],
)
