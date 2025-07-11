"""
Game agent that uses a Vision Language Model to play Super Hexagon.
Captures screen frames and provides keyboard controls based on model predictions.
"""

# Standard library imports
import _thread
import queue
import re
import time
from contextlib import contextmanager
from copy import deepcopy

# Third-party imports
import line_profiler
import torch
import typer
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from trigger_speedhack import disable_speedhack, enable_speedhack

# from transformers.models.smolvlm import SmolVLMProcessor
# Local imports
from owa.core import Runnable
from owa.core.registry import CALLABLES, LISTENERS, activate_module
from owa_game_agent.data import OWATrainingSample
from owa_game_agent.data.datasets.smolvlm2 import sample_to_smolvlm_input
from owa_game_agent.data.sample_processor import SampleProcessor

# Configuration constants
WINDOW_NAME = "hexagon"
FPS = 20
MAX_SCREEN_FRAMES = 5
MODEL_SAMPLE_DELAY = 10.0  # seconds
TIMESTAMP_INTERVAL = 0.05  # seconds
INFERENCE_TIME_LIMIT = 0.25


# Setup logger for use with tqdm
def setup_logger():
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
    logger.disable("owa.env.gst")  # suppress pipeline print


class EventManager:
    """Manages event collection and processing from various sources."""

    def __init__(self):
        self.msg_queue = queue.Queue()
        self.pbar = tqdm(desc="Recording", unit="event", dynamic_ncols=True)

    def event_callback(self, event, topic=None):
        self.msg_queue.put((topic, event, time.time_ns()))
        self.pbar.update(1)

    def keyboard_callback(self, event):
        # Exit on ESC key
        if event.vk == 0x1B:
            _thread.interrupt_main()
            exit(0)

    def screen_callback(self, event):
        self.event_callback(event, topic="screen")


class SampleManager(Runnable):
    """Manages collection and processing of input samples."""

    def on_configure(self, event_manager):
        self.queue = event_manager.msg_queue
        self._sample = OWATrainingSample(
            state_keyboard=[], state_mouse=None, state_screen=[], action_mouse=None, action_keyboard=[]
        )

    def loop(self, stop_event):
        while not stop_event.is_set():
            try:
                topic, event, publish_time = self.queue.get(timeout=1)
            except queue.Empty:
                continue

            if topic == "screen":
                frame_arr = event.frame_arr
                self._sample.state_screen.append((publish_time, frame_arr))
                if len(self._sample.state_screen) > MAX_SCREEN_FRAMES:
                    self._sample.state_screen.pop(0)

    def grab_sample(self) -> OWATrainingSample:
        """Capture current state including keyboard, mouse, and screen."""
        state_keyboard = CALLABLES["keyboard.get_state"]().buttons - {1, 2, 4}
        state_mouse = CALLABLES["mouse.get_state"]()
        self._sample.state_keyboard = state_keyboard
        self._sample.state_mouse = state_mouse
        logger.info(f"State keyboard: {state_keyboard}")
        return deepcopy(self._sample)


def logits_processor(input_ids: torch.LongTensor, scores: torch.FloatTensor):
    """Process logits to adjust model generation behavior."""
    # Adjust token probabilities as needed
    # Example (commented out):
    # scores[:, 49354] *= 0.98  # <KEYBOARD_37_0> (left)
    # scores[:, 49358] *= 0.98  # <KEYBOARD_39_0> (right)
    return scores


class Agent(Runnable):
    """AI agent that processes samples and generates actions."""

    def on_configure(self, model_id: str, sample_manager: SampleManager, event_manager: EventManager):
        self.processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            # load_in_4bit=True,
            # _attn_implementation="flash_attention_2",
        ).to("cuda")
        self.sample_manager = sample_manager
        self.event_manager = event_manager

    @line_profiler.profile
    def loop(self, stop_event):
        while not stop_event.is_set():
            sample = self.sample_manager.grab_sample()
            if len(sample.state_screen) < MAX_SCREEN_FRAMES:
                continue

            now = time.time()
            enable_speedhack()
            generated = self._generate_response(sample)
            taken = time.time() - now
            logger.info(f"Generated: {generated}, taken: {taken:.2f}s")

            disable_speedhack()
            CALLABLES["window.make_active"](WINDOW_NAME)
            taken = now  # NOTE: time stop
            self.execute(generated, anchor_time=now, processing_time=taken)

    @line_profiler.profile
    def _generate_response(self, sample):
        """Process the sample and generate a response using the VLM."""
        sample_processor = SampleProcessor()
        tokenized_sample = sample_processor.tokenize(sample)
        vlm_input = sample_to_smolvlm_input(tokenized_sample)

        example = {"messages": vlm_input.messages, "images": vlm_input.images}
        examples = [example]
        texts = []
        images = []

        for ex in examples:
            assistant_prompt = ex["messages"].pop(-1)  # noqa: F841
            texts.append(
                self.processor.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=True) + " "
            )
            images.append(ex["images"])

        # profile: 38.7%
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(
            self.model.device, dtype=self.model.dtype
        )

        # profile: 57.9%
        outputs = self.model.generate(**batch, logits_processor=[logits_processor], do_sample=False, max_new_tokens=64)

        output = outputs[0]
        generated = self.processor.decode(output, skip_special_tokens=True)
        return generated[generated.find("Assistant: ") + len("Assistant: ") :]

    def execute(self, generated: str, anchor_time: float, processing_time: float):
        """Execute the generated response as scheduled keyboard actions."""
        tokens = re.findall(r"<(.*?)>", generated)
        timestamp_list = []
        events = []

        # Parse tokens: build a list of timestamps and pair them with keyboard actions.
        for token in tokens:
            if token.startswith("TIMESTAMP"):
                timestamp = int(token.split("_")[1])
                # Each timestamp represents an absolute time: anchor_time + (timestamp * TIMESTAMP_INTERVAL)
                timestamp_list.append(anchor_time + timestamp * TIMESTAMP_INTERVAL)
            elif token.startswith("KEYBOARD"):
                if not timestamp_list:
                    logger.warning("Found KEYBOARD without TIMESTAMP")
                    time.sleep(INFERENCE_TIME_LIMIT)
                    return
                ts = timestamp_list.pop(0)
                vk, state = map(int, token.split("_")[1:])
                events.append((ts, vk, state, token))
            else:
                logger.warning(f"Invalid token: {token}")
                time.sleep(INFERENCE_TIME_LIMIT)
                return

        if not events:
            time.sleep(INFERENCE_TIME_LIMIT)
            return

        start = time.time()

        # The first event's intended time
        base_timestamp = events[0][0]  # e.g. anchor_time + 12 * 0.05 = anchor_time + 0.6

        # Compute delay needed for the first event:
        # Ideally, we want to execute the first event at base_timestamp.
        # But if processing already took processing_time, then the remaining delay is:
        first_delay = (base_timestamp - anchor_time) - processing_time
        if first_delay < 0:
            first_delay = 0

        # Schedule the first event to run after first_delay seconds from now.
        first_event_execution_time = time.time() + first_delay

        s = time.time()
        # Execute events while preserving relative differences.
        for original_time, vk, state, token in events:
            # Calculate the time difference (delta) relative to the first event.
            delta = original_time - base_timestamp
            # The scheduled time for this event is:
            scheduled_time = first_event_execution_time + delta
            to_sleep = max(0, scheduled_time - time.time())
            logger.info(f"Sleeping for {to_sleep:.2f}s, processing time {processing_time:.2f}s, token {token}")

            if time.time() - s + to_sleep > INFERENCE_TIME_LIMIT:  # if event passes time limit, break
                break

            time.sleep(to_sleep)

            # Execute the key action.
            if state:
                CALLABLES["keyboard.press"](vk)
            else:
                CALLABLES["keyboard.release"](vk)

            self.event_manager.pbar.set_description(f"Executing: {token}, {vk}, {state}")

        end = time.time()
        if end - start < INFERENCE_TIME_LIMIT:
            time.sleep(max(0, INFERENCE_TIME_LIMIT - (end - start)))


@contextmanager
def setup_resources(model_id: str):
    """Set up and tear down all resources needed for the application."""
    setup_logger()
    activate_module("owa.env.desktop")
    activate_module("owa.env.gst")

    event_manager = EventManager()
    recorder = LISTENERS["screen"]().configure(
        window_name=WINDOW_NAME, fps=FPS, callback=event_manager.screen_callback
    )
    keyboard_listener = LISTENERS["keyboard"]().configure(callback=event_manager.keyboard_callback)
    sample_manager = SampleManager().configure(event_manager=event_manager)
    agent = Agent().configure(model_id=model_id, sample_manager=sample_manager, event_manager=event_manager)

    resources = [
        (recorder, "recorder"),
        (keyboard_listener, "keyboard listener"),
        (sample_manager, "sample manager"),
        (agent, "agent"),
    ]

    # Start all resources
    for resource, name in resources:
        resource.start()
        logger.info(f"Started {name}")

    try:
        yield
    finally:
        # Stop all resources in reverse order
        for resource, name in reversed(resources):
            try:
                resource.stop()
                resource.join(timeout=5)
                logger.info(f"Stopped {name}")
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")


def main(model_id: str):
    """Run the game agent with the specified model."""
    with setup_resources(model_id):
        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Operation stopped by user.")


if __name__ == "__main__":
    typer.run(main)
