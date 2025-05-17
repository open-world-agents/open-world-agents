from typing import Any

import cv2
from PIL import Image
from pydantic import BaseModel

from owa.agent.core.perception import Perception, PerceptionSpecDict, apply_spec
from owa.env.gst.msg import FrameStamped, ScreenEmitted

CHAT_INSTRUCTION = """You are playing Super Hexagon, a fast-paced game that requires precise control and timing.
After this prompt, you will receive {len_images} sequential image frames that show the game's visual history from the past to the present.
Using the current keyboard state and the image sequence, predict the future sequence of keyboard actions. For each action, include the timestamp when it should be executed."""


class SmolVLMInput(BaseModel):
    images: list[Any]
    messages: list[dict]


def perception_to_conversation(
    perception_history: Perception, current_perception: Perception, *, now: int, spec: PerceptionSpecDict
) -> tuple[Perception, dict]:
    """For events later than 'now', it's considered as future events('label')."""
    perception_history += current_perception
    perception_history, info = apply_spec(perception_history, now=now, spec=spec)
    if perception_history:
        items = "Summary of Perception History:\n"
        for channel, events in perception_history.items():
            # items.append(channel, len(events))
            items += f"{channel}: {len(events)} events\n"

        images = perception_history["inputs/screen"]
        len_images = len(images)

        assistant_message = sorted(
            perception_history["outputs/keyboard"] + perception_history["outputs/mouse"], key=lambda x: x.timestamp
        )
        assistant_message = str(assistant_message)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CHAT_INSTRUCTION.format(len_images=len_images) + "<image>" * len_images},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_message},
                ],
            },
        ]
        conversation = SmolVLMInput(images=images, messages=messages)
    else:
        conversation = None
    return perception_history, conversation


def lazy_load_images(conversation):
    images = []
    for event in conversation.images:
        if isinstance(event, FrameStamped):
            screen = event.msg.frame_arr
            screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2RGB)
            images.append(Image.fromarray(screen))
        elif isinstance(event, ScreenEmitted):
            images.append(event.to_pil_image())
    conversation.images = images
    return conversation


def apply_processor(inputs: SmolVLMInput, *, processor):
    # Placeholder for the actual processor application logic
    return {"input_ids": inputs.messages[0]["content"][0]["text"]}
