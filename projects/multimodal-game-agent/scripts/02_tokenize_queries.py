"""
This script's I/O

Input: list[query]
Output: N/A
"""

from pathlib import Path
from typing import Any, List, Tuple

import typer
from PIL import Image
from pydantic import BaseModel

from owa.env.desktop.constants import VK
from owa_game_agent.data import OWAMcapQuery, OWATrainingSample
from owa.env.gst.msg import ScreenEmitted
app = typer.Typer()


def convert_timestamp_to_token(timestamp: int) -> str:
    """
    Convert a nanosecond timestamp to a discrete token.
    The rule here is to divide the timestamp by 10^7 and multiply by 10, which is timestamp token is discretize by each 0.01 second.
    Also, 5000000ns is added to the timestamp to round up to the nearest 0.01 second.
    For example, 274866265ns becomes 2750_MILLISECOND, denoted as 2750_MS.
    """
    token_val = (timestamp + 5000000) // 10000000 * 10
    return f"{token_val}_MS"


def get_vk_name(code: int) -> str:
    """
    Attempt to retrieve the enum member name from the VK enum given a key code.
    Returns None if the code is not in VK.
    """
    try:
        return VK(code).name
    except ValueError:
        return None


def process_state_keyboard(state_keyboard: List[dict]) -> List[str]:
    """
    Process state_keyboard events into discrete tokens.
    Each dictionary in state_keyboard should have a 'pressed_vk_list' key.
    For each key code in that list, we generate a token like "KEYBOARD_A_PRESSED"
    if the key is in our mapping.
    """
    tokens = []
    for entry in state_keyboard:
        pressed_vk_list = entry.get("pressed_vk_list", [])
        for vk in pressed_vk_list:
            key_letter = get_vk_name(vk)
            if key_letter:
                token = f"KEYBOARD_{key_letter.upper()}_PRESSED"
                tokens.append(token)
    return tokens


def process_action_keyboard(action_keyboard: List[Tuple[int, dict]]) -> List[str]:
    """
    Process action_keyboard events into discrete tokens.
    Each event is a tuple of (timestamp, event_info).
    The event_info dict should contain 'event_type' (press/release) and 'vk'.
    We also convert the timestamp to a discrete token.
    """
    tokens = []
    for event in action_keyboard:
        timestamp, event_info = event
        # Convert timestamp
        time_token = convert_timestamp_to_token(timestamp)
        # Process keyboard event
        event_type = event_info.get("event_type")
        vk = event_info.get("vk")
        vk_name = get_vk_name(vk)
        if vk_name and event_type in {"press", "release"}:
            # Map 'press' -> PRESSED, 'release' -> RELEASED
            event_str = "PRESSED" if event_type == "press" else "RELEASED"
            key_token = f"KEYBOARD_{vk_name}_{event_str}"
            tokens.append(time_token)
            tokens.append(key_token)
        else:
            print(f"Skip Invalid event or key_token: {event_type}, {vk}")
    return tokens


def rule_based_tokenize_sample(sample: OWATrainingSample) -> List[str]:
    """
    Tokenize the given sample using rule-based tokenization for keyboard events.
    Processes only state_keyboard and action_keyboard.
    """
    sample.state_keyboard = process_state_keyboard(sample.state_keyboard)
    sample.action_keyboard = process_action_keyboard(sample.action_keyboard)
    return sample


class SmolVLMInput(BaseModel):
    images: List[Any]
    messages: List[dict]

    class Config:
        arbitrary_types_allowed = True


def sample_to_smolvlm_input(sample: OWATrainingSample) -> SmolVLMInput:
    CHAT_INSTRUCTION = """You are playing Super Hexagon, a fast-paced game that requires precise control and timing. The current keyboard state, which represents the keys that are pressed, is {keyboard_state}.
After this prompt, you will receive {len_images} sequential image frames that show the game’s visual history from the past to the present.
Using the current keyboard state and the image sequence, predict the future sequence of keyboard actions. For each action, include the timestamp when it should be executed."""

    keyboard_state = sample.state_keyboard
    if keyboard_state is None:
        keyboard_state = "None"
    keyboard_action = sample.action_keyboard
    state_screen = sample.state_screen

    state_screen = [ScreenEmitted(**screen).to_pil_image() for screen in state_screen]

    len_images = len(state_screen)

    # It will be processed like [keyboard1, keyboard2, ...], -> keyboard1keyboard2...
    keyboard_state = "".join(str(item) for item in keyboard_state)
    # It will be processed like [(timestamp1, keyboard1), (timestamp2, keyboard2), ...], -> timestamp1keyboard1timestamp2keyboard2...
    keyboard_action = "".join(str(item) for pair in keyboard_action for item in pair)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": CHAT_INSTRUCTION.format_map({"len_images": len_images, "keyboard_state": keyboard_state})
                    + "<image>" * len_images,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": keyboard_action},
            ],
        },
    ]

    return SmolVLMInput(images=state_screen, messages=messages)


@app.command()
def main(query_path: Path):
    # for each query, extract the sample.
    # query_path is jsonl file
    with open(query_path, "r") as f:
        queries = [OWAMcapQuery.model_validate_json(line) for line in f]

    # TODO: implement following
    query = queries[0]
    sample = query.to_sample()
    print(sample)
    print("==============")

    sample = rule_based_tokenize_sample(sample)
    print(sample)
    print("==============")

    sample = sample_to_smolvlm_input(sample)
    print(sample)
    print("==============")


if __name__ == "__main__":
    app()
