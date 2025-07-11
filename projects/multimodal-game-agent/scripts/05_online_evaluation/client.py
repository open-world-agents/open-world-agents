"""
Game agent client that captures screen frames and sends them to a remote model server.
"""

# Standard library imports
import base64
import io
import queue
import re
import time
from contextlib import contextmanager
from copy import deepcopy

# Third-party imports
import numpy as np
import requests
import typer
from loguru import logger
from PIL import Image
from tqdm import tqdm

# Local imports
from owa.core import Runnable
from owa.core.registry import CALLABLES, LISTENERS, activate_module
from owa_game_agent.data import OWATrainingSample
from owa_game_agent.data.datasets.smolvlm2 import sample_to_smolvlm_input
from owa_game_agent.data.sample_processor import SampleProcessor

# Configuration constants
WINDOW_NAME = "Super Hexagon"
FPS = 5
MAX_SCREEN_FRAMES = 5
MODEL_SAMPLE_DELAY = 0.5  # seconds
MODEL_SERVER_URL = "http://your-server-ip:8000/api/inference"  # Update with your server address


# Setup logger for use with tqdm
def setup_logger():
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
    logger.disable("owa.env.gst")  # suppress pipeline print


def encode_image(image_array: np.ndarray) -> str:
    """Encode numpy array image to base64 string for API transmission."""
    img = Image.fromarray(image_array)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


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
            exit()

    def screen_callback(self, event):
        self.event_callback(event, topic="screen")


class SampleManager(Runnable):
    """Manages collection and processing of input samples."""

    def on_configure(self, event_manager):
        self.event_manager = event_manager
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


class RemoteAgent(Runnable):
    """Agent that sends samples to a remote model server and executes received actions."""

    def on_configure(self, sample_manager: SampleManager, event_manager: EventManager):
        self.sample_manager = sample_manager
        self.event_manager = event_manager
        self.session = requests.Session()
        # Test connection to server
        try:
            self.session.get(MODEL_SERVER_URL.replace("/api/inference", "/docs"))
            logger.info(f"Successfully connected to model server at {MODEL_SERVER_URL}")
        except requests.RequestException as e:
            logger.error(f"Failed to connect to model server: {e}")
            logger.warning("The agent will still run, but model inference will fail until server is available")

    def loop(self, stop_event):
        while not stop_event.is_set():
            sample = self.sample_manager.grab_sample()
            if len(sample.state_screen) < MAX_SCREEN_FRAMES:
                continue

            now = time.time()
            try:
                generated = self._get_remote_inference(sample)
                logger.info(f"Generated: {generated}, taken: {time.time() - now:.2f}s")

                if CALLABLES["window.is_active"](WINDOW_NAME):
                    self.execute(generated, anchor_time=now)
            except Exception as e:
                logger.error(f"Error in remote inference: {e}")
                time.sleep(1)  # Avoid rapid retries on failure

    def _get_remote_inference(self, sample):
        """Send sample to remote model server and get response."""
        sample_processor = SampleProcessor()
        tokenized_sample = sample_processor.tokenize(sample)
        vlm_input = sample_to_smolvlm_input(tokenized_sample)

        # Encode images for transmission
        encoded_images = []
        for _, frame in sample.state_screen:
            encoded_images.append(encode_image(frame))

        # Prepare request payload
        payload = {"messages": vlm_input.messages, "images": encoded_images}

        # Send request to server
        response = self.session.post(MODEL_SERVER_URL, json=payload)
        response.raise_for_status()
        result = response.json()

        return result["generated_text"]

    def execute(self, generated, anchor_time: float):
        """Execute the generated response as keyboard actions."""
        tokens = re.findall(r"<(.*?)>", generated)
        sum_time = anchor_time

        for token in tokens:
            if token.startswith("TIMESTAMP"):
                timestamp = int(token.split("_")[1])
                sum_time = anchor_time + timestamp * 0.05
                to_sleep = max(0, sum_time - time.time())
                if sum_time + to_sleep - anchor_time > MODEL_SAMPLE_DELAY:
                    break
                time.sleep(to_sleep)
            elif token.startswith("KEYBOARD"):
                vk, state = map(int, token.split("_")[1:])
                if state:
                    CALLABLES["keyboard.press"](vk)
                else:
                    CALLABLES["keyboard.release"](vk)
                self.event_manager.pbar.set_description(f"Executing: {token}, {vk}, {state}")
            else:
                logger.warning(f"Invalid token: {token}")


@contextmanager
def setup_resources():
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
    agent = RemoteAgent().configure(sample_manager=sample_manager, event_manager=event_manager)

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


def main():
    """Run the game agent client."""
    with setup_resources():
        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Operation stopped by user.")


if __name__ == "__main__":
    typer.run(main)
