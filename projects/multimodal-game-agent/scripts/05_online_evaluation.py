import argparse
import re
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from queue import Queue
from typing import List, Optional

import cv2
import line_profiler
import numpy as np
import pandas as pd
import torch
import typer
from loguru import logger
from pydantic import BaseModel
from torch.utils.data import Subset
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, GPT2Tokenizer, GPT2TokenizerFast
from transformers.models.smolvlm import SmolVLMProcessor
from typing_extensions import Annotated

from mcap_owa.highlevel import OWAMcapWriter
from owa.core import Runnable
from owa.core.registry import CALLABLES, LISTENERS, activate_module
from owa.core.time import TimeUnits
from owa_game_agent.data import OWAMcapQuery, OWATrainingSample
from owa_game_agent.data.datasets.smolvlm2 import SmolVLM2Dataset, collate_fn, sample_to_smolvlm_input
from owa_game_agent.data.sample_processor import (
    KEYBOARD_EVENT_TOKEN_FORMAT,
    KEYBOARD_STATE_COUNT,
    KEYBOARD_VK_COUNT,
    TIMESTAMP_TOKEN_COUNT,
    TIMESTAMP_TOKEN_FORMAT,
    SampleProcessor,
)

# how to use loguru with tqdm: https://github.com/Delgan/loguru/issues/135
logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

# TODO: apply https://loguru.readthedocs.io/en/stable/resources/recipes.html#configuring-loguru-to-be-used-by-a-library-or-an-application
logger.disable("owa.env.gst")  # suppress pipeline print

queue = Queue()
pbar = tqdm(desc="Recording", unit="event", dynamic_ncols=True)


def callback(event, topic=None):
    queue.put((topic, event, time.time_ns()))


def keyboard_publisher_callback(event):
    # exit for esc
    if event.vk == 0x1B:
        exit()


def screen_publisher_callback(event):
    callback(event, topic="screen")


def configure():
    activate_module("owa.env.desktop")
    activate_module("owa.env.gst")


class SampleManager(Runnable):
    def on_configure(self, queue: Queue):
        self.queue = queue
        self._sample = OWATrainingSample(
            state_keyboard=[], state_mouse=None, state_screen=[], action_mouse=None, action_keyboard=[]
        )

    def loop(self, stop_event):
        while not stop_event.is_set():
            topic, event, publish_time = self.queue.get()
            pbar.update(1)

            if topic == "screen":
                frame_arr = event.frame_arr
                self._sample.state_screen.append((publish_time, frame_arr))
                if len(self._sample.state_screen) > 5:
                    self._sample.state_screen.pop(0)

    def grab_sample(self) -> OWATrainingSample:
        state_keyboard = CALLABLES["keyboard.get_state"]().buttons - {1, 2, 4}
        state_mouse = CALLABLES["mouse.get_state"]()
        self._sample.state_keyboard = state_keyboard
        self._sample.state_mouse = state_mouse
        logger.info(f"State keyboard: {state_keyboard}")
        return deepcopy(self._sample)


def logits_processor(input_ids: torch.LongTensor, scores: torch.FloatTensor):
    # scores[:, 49418] *= 0.8  # <KEYBOARD_69_0>
    # scores[:, 49419] *= 0.8  # <KEYBOARD_69_1>
    return scores


class Agent(Runnable):
    def on_configure(self, model_id: str, sample_manager: SampleManager):
        self.processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            # _attn_implementation="flash_attention_2",
        ).to("cuda")
        self.sample_manager = sample_manager

    def loop(self, stop_event):
        while not stop_event.is_set():
            sample = self.sample_manager.grab_sample()
            if len(sample.state_screen) < 5:
                continue

            sample_processor = SampleProcessor()
            tokenized_sample = sample_processor.tokenize(sample)
            vlm_input = sample_to_smolvlm_input(tokenized_sample)
            example = {"messages": vlm_input.messages, "images": vlm_input.images}

            examples = [example]
            texts = []
            labels = []
            images = []

            for ex in examples:
                assistant_prompt = ex["messages"].pop(-1)
                # BUG: Surprisingly, SmolVLM processor does NOT append "Assistant: " as generation prompt, but append "Assistant:"!
                # SEVERE bug because single space can be critical for the model to generate the correct response.
                texts.append(
                    self.processor.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=True)
                    + " "
                )
                labels.append(assistant_prompt["content"][0]["text"])
                images.append(ex["images"])

            batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(
                self.model.device, dtype=self.model.dtype
            )

            outputs = self.model.generate(
                **batch, logits_processor=[logits_processor], do_sample=False, max_new_tokens=64
            )
            output = outputs[0]

            generated = self.processor.decode(output, skip_special_tokens=True)
            generated = generated[generated.find("Assistant: ") + len("Assistant: ") :]

            logger.info(f"Generated: {generated}")

            if CALLABLES["window.is_active"]("ZType"):
                self.execute(generated)

    def execute(self, generated):
        """Execute the generated response.
        generated example: <TIMESTAMP_0><KEYBOARD_69_0><TIMESTAMP_0><KEYBOARD_69_0><TIMESTAMP_2><KEYBOARD_69_0><TIMESTAMP_3><KEYBOARD_69_0><TIMESTAMP_4><KEYBOARD_69_0><TIMESTAMP_4><KEYBOARD_69_0><TIMESTAMP_6><KEYBOARD_69_0><TIMESTAMP_7><KEYBOARD_69_0><TIMESTAMP_7><KEYBOARD_69_0><TIMESTAMP_7><KEYBOARD_69_0><TIMESTAMP_8><KEYBOARD_69_0><TIMESTAMP_9><KEYBOARD_69_0><TIMESTAMP_10><KEYBOARD_69_0>
        TIMESTAMP_0: 0 seconds, TIMESTAMP_i: i * 50 ms
        KEYBOARD_i_{0/1}: press(1)/release(0) virtual key code i
        """
        tokens = re.findall(r"<(.*?)>", generated)
        total_time = 0

        for token in tokens:
            # limit the execution time to 0.25 seconds
            if total_time > 0.25:
                break
            if token.startswith("TIMESTAMP"):
                timestamp = int(token.split("_")[1])
                time.sleep(timestamp * 0.05)
                total_time += timestamp * 0.05
            elif token.startswith("KEYBOARD"):
                vk, state = map(int, token.split("_")[1:])
                # CALLABLES["keyboard.press"](vk) if state else CALLABLES["keyboard.release"](vk)
                CALLABLES["keyboard.press"](vk), CALLABLES["keyboard.release"](vk)
                pbar.set_description(f"Executing: {token}, {vk}, {state}")
            else:
                raise ValueError(f"Invalid token: {token}")


def main(model_id: str):
    """Record screen, keyboard, mouse, and window events to an `.mcap` and `.mkv` file."""

    configure()
    recorder = LISTENERS["screen"]().configure(window_name="ZType", fps=5, callback=screen_publisher_callback)
    keyboard_listener = LISTENERS["keyboard"]().configure(callback=keyboard_publisher_callback)
    sample_manager = SampleManager().configure(queue=queue)
    agent = Agent().configure(model_id=model_id, sample_manager=sample_manager)

    try:
        recorder.start()
        keyboard_listener.start()
        sample_manager.start()
        agent.start()

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Recording stopped by user.")
    finally:
        # resource cleanup

        try:
            recorder.stop()
            recorder.join(timeout=5)
        except Exception as e:
            logger.error(f"Error occurred while stopping the recorder: {e}")

        try:
            keyboard_listener.stop()
            keyboard_listener.join(timeout=5)
        except Exception as e:
            logger.error(f"Error occurred while stopping the listeners: {e}")

        try:
            sample_manager.stop()
            sample_manager.join(timeout=5)
        except Exception as e:
            logger.error(f"Error occurred while stopping the sample manager: {e}")

        try:
            agent.stop()
            agent.join(timeout=5)
        except Exception as e:
            logger.error(f"Error occurred while stopping the agent: {e}")


if __name__ == "__main__":
    typer.run(main)
