import queue
import re
import time
from copy import deepcopy

import line_profiler
import torch
import typer
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from owa.core import Runnable
from owa.core.registry import CALLABLES, LISTENERS, activate_module
from owa_game_agent.data import OWATrainingSample
from owa_game_agent.data.datasets.smolvlm2 import sample_to_smolvlm_input
from owa_game_agent.data.sample_processor import SampleProcessor

# how to use loguru with tqdm: https://github.com/Delgan/loguru/issues/135
logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

# TODO: apply https://loguru.readthedocs.io/en/stable/resources/recipes.html#configuring-loguru-to-be-used-by-a-library-or-an-application
logger.disable("owa.env.gst")  # suppress pipeline print

msg_queue = queue.Queue()
pbar = tqdm(desc="Recording", unit="event", dynamic_ncols=True)
WINDOW_NAME = "Super Hexagon"


def callback(event, topic=None):
    msg_queue.put((topic, event, time.time_ns()))


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
    def on_configure(self, queue: queue.Queue):
        self.queue = queue
        self._sample = OWATrainingSample(
            state_keyboard=[], state_mouse=None, state_screen=[], action_mouse=None, action_keyboard=[]
        )

    def loop(self, stop_event):
        while not stop_event.is_set():
            try:
                topic, event, publish_time = self.queue.get(timeout=1)
            except queue.Empty:
                continue
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
    # # left
    # scores[:, 49354] *= 0.98  # <KEYBOARD_37_0>
    # # right
    # scores[:, 49358] *= 0.98  # <KEYBOARD_39_0>
    # print(scores[:, 49354:49359])

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

    @line_profiler.profile
    def loop(self, stop_event):
        while not stop_event.is_set():
            sample = self.sample_manager.grab_sample()
            if len(sample.state_screen) < 5:
                continue

            now = time.time()
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

            # profile: 48ms, 24.1%
            batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(
                self.model.device, dtype=self.model.dtype
            )

            # profile: 77ms, 57.5%
            outputs = self.model.generate(
                **batch, logits_processor=[logits_processor], do_sample=False, max_new_tokens=64
            )
            output = outputs[0]

            generated = self.processor.decode(output, skip_special_tokens=True)
            generated = generated[generated.find("Assistant: ") + len("Assistant: ") :]

            logger.info(f"Generated: {generated}, taken: {time.time() - now:.2f}s")

            if CALLABLES["window.is_active"](WINDOW_NAME):
                self.execute(generated, anchor_time=now)

    def execute(self, generated, anchor_time: float):
        """Execute the generated response.
        generated example: <TIMESTAMP_0><KEYBOARD_69_0><TIMESTAMP_0><KEYBOARD_69_0><TIMESTAMP_2><KEYBOARD_69_0><TIMESTAMP_3><KEYBOARD_69_0><TIMESTAMP_4><KEYBOARD_69_0><TIMESTAMP_4><KEYBOARD_69_0><TIMESTAMP_6><KEYBOARD_69_0><TIMESTAMP_7><KEYBOARD_69_0><TIMESTAMP_7><KEYBOARD_69_0><TIMESTAMP_7><KEYBOARD_69_0><TIMESTAMP_8><KEYBOARD_69_0><TIMESTAMP_9><KEYBOARD_69_0><TIMESTAMP_10><KEYBOARD_69_0>
        TIMESTAMP_0: 0 seconds, TIMESTAMP_i: i * 50 ms
        KEYBOARD_i_{0/1}: press(1)/release(0) virtual key code i
        """
        tokens = re.findall(r"<(.*?)>", generated)
        sum_time = anchor_time

        for token in tokens:
            if token.startswith("TIMESTAMP"):
                timestamp = int(token.split("_")[1])
                sum_time = anchor_time + timestamp * 0.05
                to_sleep = max(0, sum_time - time.time())
                if sum_time + to_sleep - anchor_time > 0.50:
                    break
                time.sleep(to_sleep)
            elif token.startswith("KEYBOARD"):
                vk, state = map(int, token.split("_")[1:])
                CALLABLES["keyboard.press"](vk) if state else CALLABLES["keyboard.release"](vk)
                pbar.set_description(f"Executing: {token}, {vk}, {state}")
            else:
                raise ValueError(f"Invalid token: {token}")


def main(model_id: str):
    """Record screen, keyboard, mouse, and window events to an `.mcap` and `.mkv` file."""

    configure()
    recorder = LISTENERS["screen"]().configure(window_name=WINDOW_NAME, fps=5, callback=screen_publisher_callback)
    keyboard_listener = LISTENERS["keyboard"]().configure(callback=keyboard_publisher_callback)
    sample_manager = SampleManager().configure(queue=msg_queue)
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
            logger.info("Stopped the recorder.")
        except Exception as e:
            logger.error(f"Error occurred while stopping the recorder: {e}")

        try:
            keyboard_listener.stop()
            keyboard_listener.join(timeout=5)
            logger.info("Stopped the keyboard listener.")
        except Exception as e:
            logger.error(f"Error occurred while stopping the listeners: {e}")

        try:
            sample_manager.stop()
            sample_manager.join(timeout=5)
            logger.info("Stopped the sample manager.")
        except Exception as e:
            logger.error(f"Error occurred while stopping the sample manager: {e}")

        try:
            agent.stop()
            agent.join(timeout=5)
            logger.info("Stopped the agent.")
        except Exception as e:
            logger.error(f"Error occurred while stopping the agent: {e}")


if __name__ == "__main__":
    typer.run(main)
