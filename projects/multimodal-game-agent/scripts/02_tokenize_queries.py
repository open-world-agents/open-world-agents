"""
This script's I/O

Input: list[query]
Output: N/A
"""

from pathlib import Path
from typing import Any, List, Tuple

import typer
from loguru import logger
from pydantic import BaseModel

from owa.core.time import TimeUnits
from owa.env.gst.msg import ScreenEmitted
from owa_game_agent.data import OWAMcapQuery, OWATrainingSample

# Constants
TIMESTAMP_MIN_NS = 0
TIMESTAMP_MAX_NS = TimeUnits.SECOND
TIMESTAMP_TOKEN_INTERVAL_NS = TimeUnits.MSECOND * 10

MS_TOKEN_FORMAT = "<TIMESTAMP_{}>"
KEYBOARD_EVENT_TOKEN_FORMAT = "<KEYBOARD_{}_{}>>"
CHAT_INSTRUCTION = """You are playing Super Hexagon, a fast-paced game that requires precise control and timing. The current keyboard state, which represents the keys that are pressed, is {keyboard_state}.
After this prompt, you will receive {len_images} sequential image frames that show the game's visual history from the past to the present.
Using the current keyboard state and the image sequence, predict the future sequence of keyboard actions. For each action, include the timestamp when it should be executed."""

app = typer.Typer()


class TokenizationHelper:
    @staticmethod
    def convert_timestamp_to_token(timestamp: int) -> str:
        if not TIMESTAMP_MIN_NS <= timestamp <= TIMESTAMP_MAX_NS:
            raise ValueError(f"Invalid timestamp: {timestamp}")

        index = (timestamp - TIMESTAMP_MIN_NS) // TIMESTAMP_TOKEN_INTERVAL_NS
        return MS_TOKEN_FORMAT.format(index)

    @staticmethod
    def process_state_keyboard(state_keyboard: List[dict]) -> List[str]:
        return [KEYBOARD_EVENT_TOKEN_FORMAT.format(pressed_key, "PRESSED") for pressed_key in state_keyboard]

    @staticmethod
    def process_action_keyboard(action_keyboard: List[Tuple[int, dict]]) -> List[str]:
        tokens = []
        for timestamp, event_info in action_keyboard:
            # Convert timestamp
            time_token = TokenizationHelper.convert_timestamp_to_token(timestamp)

            # Process keyboard event
            event_type = event_info.get("event_type")
            vk = event_info.get("vk")

            if event_type in {"press", "release"}:
                # Map 'press' -> PRESSED, 'release' -> RELEASED
                event_str = "PRESSED" if event_type == "press" else "RELEASED"
                key_token = KEYBOARD_EVENT_TOKEN_FORMAT.format(vk, event_str)
                tokens.extend([time_token, key_token])
            else:
                print(f"Skip Invalid event or key_token: {event_type}, {vk}")

        return tokens


class SmolVLMInput(BaseModel):
    images: list[Any]
    messages: list[dict]


class SampleProcessor:
    @staticmethod
    def tokenize_sample(sample: OWATrainingSample) -> OWATrainingSample:
        """
        Tokenize the given sample using rule-based tokenization for keyboard events.
        Processes only state_keyboard and action_keyboard.
        """
        tokenized_sample = sample.model_copy(deep=True)
        tokenized_sample.state_keyboard = TokenizationHelper.process_state_keyboard(sample.state_keyboard)
        tokenized_sample.action_keyboard = TokenizationHelper.process_action_keyboard(sample.action_keyboard)
        return tokenized_sample

    @staticmethod
    def to_smolvlm_input(sample: OWATrainingSample) -> SmolVLMInput:
        """
        Convert a training sample to SmolVLM input format with images and messages.
        """
        # Convert state_keyboard to string representation
        keyboard_state = (
            "None" if sample.state_keyboard is None else "".join(str(item) for item in sample.state_keyboard)
        )

        # Convert action_keyboard to string representation
        keyboard_action = "".join(str(item) for pair in sample.action_keyboard for item in pair)

        # Convert screen states to PIL images
        state_screen = [ScreenEmitted(**screen).to_pil_image() for timestamp, screen in sample.state_screen]
        len_images = len(state_screen)

        # Create messages for SmolVLM
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": CHAT_INSTRUCTION.format(len_images=len_images, keyboard_state=keyboard_state)
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


def load_queries(query_path: Path) -> List[OWAMcapQuery]:
    """Load queries from a JSONL file."""
    with open(query_path, "r") as f:
        return [OWAMcapQuery.model_validate_json(line) for line in f]


@app.command()
def main(query_path: Path):
    logger.info(
        f"Total timestamp tokens: {(TIMESTAMP_MAX_NS - TIMESTAMP_MIN_NS) // TIMESTAMP_TOKEN_INTERVAL_NS}, "
        f"ranging from {TIMESTAMP_MIN_NS / TimeUnits.SECOND} to {TIMESTAMP_MAX_NS / TimeUnits.SECOND} seconds"
    )

    # Load all queries from the file
    queries = load_queries(query_path)

    # Select a sample query for processing (same as original)
    query_index = len(queries) // 2 + 55
    query = queries[query_index]

    # Process the sample
    sample = query.to_sample()
    print(sample)
    print("==============")

    tokenized_sample = SampleProcessor.tokenize_sample(sample)
    print(tokenized_sample)
    print("==============")

    vlm_input = SampleProcessor.to_smolvlm_input(tokenized_sample)
    print(vlm_input)
    print("==============")


if __name__ == "__main__":
    app()
