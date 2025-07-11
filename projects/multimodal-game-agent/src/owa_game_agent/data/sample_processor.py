from typing import List, Tuple

from owa.core.time import TimeUnits
from owa_game_agent.data import OWATrainingSample

# Constants
TIMESTAMP_MIN_NS = 0  # 0 seconds. TODO: minus timestamp to support timestamp to state.
TIMESTAMP_MAX_NS = TimeUnits.SECOND * 2  # 2 seconds
TIMESTAMP_TOKEN_INTERVAL_NS = TimeUnits.MSECOND * 50  # 20.0 Hz
# 2 seconds * 20 Hz = 80 tokens
TIMESTAMP_TOKEN_COUNT = (TIMESTAMP_MAX_NS - TIMESTAMP_MIN_NS) // TIMESTAMP_TOKEN_INTERVAL_NS
TIMESTAMP_TOKEN_FORMAT = "<TIMESTAMP_{}>"

KEYBOARD_VK_COUNT = 256
KEYBOARD_STATE_COUNT = 2
KEYBOARD_EVENT_TOKEN_FORMAT = "<KEYBOARD_{}_{}>"


class TokenizationHelper:
    @staticmethod
    def convert_timestamp_to_token(timestamp: int) -> str:
        if not TIMESTAMP_MIN_NS <= timestamp <= TIMESTAMP_MAX_NS:
            raise ValueError(f"Invalid timestamp: {timestamp}")

        index = (timestamp - TIMESTAMP_MIN_NS) // TIMESTAMP_TOKEN_INTERVAL_NS
        return TIMESTAMP_TOKEN_FORMAT.format(index)

    @staticmethod
    def process_state_keyboard(state_keyboard: List[dict]) -> List[str]:
        return [KEYBOARD_EVENT_TOKEN_FORMAT.format(pressed_key, 1) for pressed_key in state_keyboard]

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
                key_token = KEYBOARD_EVENT_TOKEN_FORMAT.format(vk, int(event_type == "press"))
                tokens.extend([time_token, key_token])
            else:
                raise ValueError(f"Invalid event type: {event_type}")

        return tokens


class SampleProcessor:
    def tokenize(self, sample: OWATrainingSample) -> OWATrainingSample:
        """
        Tokenize the given sample using rule-based tokenization for keyboard events.
        Processes only state_keyboard and action_keyboard.
        """
        tokenized_sample = sample.model_copy(deep=True)
        tokenized_sample.state_keyboard = TokenizationHelper.process_state_keyboard(sample.state_keyboard)
        tokenized_sample.action_keyboard = TokenizationHelper.process_action_keyboard(sample.action_keyboard)
        return tokenized_sample

    def detokenize(self, tokenized_sample: OWATrainingSample) -> OWATrainingSample:
        """
        Detokenize the given sample by converting keyboard tokens back to state_keyboard and action_keyboard.
        """
        detokenized_sample = tokenized_sample.model_copy(deep=True)
        detokenized_sample.state_keyboard = self.detokenize_state_keyboard(tokenized_sample.state_keyboard)
        detokenized_sample.action_keyboard = self.detokenize_action_keyboard(tokenized_sample.action_keyboard)
        return detokenized_sample
