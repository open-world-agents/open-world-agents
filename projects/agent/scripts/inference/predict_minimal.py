import copy
import sys
from typing import Callable, Iterator, cast

import torch
from loguru import logger
from tqdm import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    PreTrainedTokenizer,
)
from transformers.cache_utils import Cache, StaticCache

from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter
from mcap_owa.highlevel.mcap_msg import McapMessage
from owa.data.datasets import load_from_disk
from owa.data.encoders import EventEncoderError, HierarchicalEventEncoder
from owa.data.episode_tokenizer import EpisodeTokenizer, TokenizedEvent
from owa.data.processing.resampler import create_resampler

logger.remove()
logger.add(sys.stderr, level="INFO")

torch_device = "cuda:0"


pretrained_model_name_or_path = "/mnt/harbor/projects/owa/checkpoints/gidm/InternVL3-1B-hf_gidm-100ms-batch128-lr2e5"
model = AutoModelForImageTextToText.from_pretrained(
    pretrained_model_name_or_path, device_map=torch_device, torch_dtype=torch.bfloat16
)
processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path)
tokenizer = processor.tokenizer

episode_tokenizer = EpisodeTokenizer.from_transformers_model(pretrained_model_name_or_path)
episode_tokenizer.prepare_model(tokenizer=tokenizer)
assert isinstance(episode_tokenizer.encoder, HierarchicalEventEncoder), "Only hierarchical event encoder is supported"


def input_stream() -> Iterator[tuple[McapMessage, TokenizedEvent]]:
    prev_msg = None
    resamplers = {
        "screen": create_resampler("screen", min_interval_ns=50_000_000),
        "mouse/raw": create_resampler("mouse/raw", min_interval_ns=50_000_000),
        "keyboard": create_resampler("keyboard", min_interval_ns=0),
    }
    with OWAMcapReader(
        "/mnt/raid12/datasets/owa_game_dataset_filtered_448/milkclouds00@gmail.com/apex_legends/0805_01.mcap"
    ) as reader:
        mid_time = reader.start_time + (reader.end_time - reader.start_time) // 2
        for mcap_msg in reader.iter_messages(topics=["screen", "mouse/raw", "keyboard"], start_time=mid_time):
            resampler = resamplers[mcap_msg.topic]
            resampler.add_event(mcap_msg)

            for mcap_msg in resampler.pop_event():
                if mcap_msg.topic != "screen":
                    logger.warning(f"Ground truth event: {mcap_msg.topic=}, {mcap_msg.timestamp=}, {mcap_msg.decoded}")
                    continue
                is_first = prev_msg is None
                prev_msg = mcap_msg

                tokenized_event = episode_tokenizer.tokenize_event(mcap_msg, is_first=is_first, is_last=False)
                # resolve relative path
                tokenized_event["images"] = [
                    img.resolve_relative_path(
                        "/mnt/raid12/datasets/owa_game_dataset_filtered_448/milkclouds00@gmail.com/apex_legends/0805_01.mcap"
                    )
                    for img in tokenized_event["images"]
                ]

                yield mcap_msg, tokenized_event


class ContextManager:
    def __init__(self, *, device, callback: Callable[[McapMessage], None] | None = None):
        self.device = device
        self.callback = callback
        # sequences containing all events
        self.sequences = torch.tensor([], dtype=torch.long, device=self.device)
        self.pixel_values = torch.tensor([], dtype=torch.bfloat16, device=self.device)
        # indices of event start tokens
        self.event_indices = torch.tensor([], dtype=torch.long, device=self.device)
        self.image_count_per_event = torch.tensor([], dtype=torch.long, device=self.device)
        self.timestamps = torch.tensor([], dtype=torch.long, device=self.device)
        self.timestamp_biases = torch.tensor([], dtype=torch.long, device=self.device)

    def __repr__(self):
        return (
            f"ContextManager(sequences={self.sequences.shape}, pixel_values={self.pixel_values.shape}, "
            f"event_indices={self.event_indices.shape}, image_count_per_event={self.image_count_per_event.shape}, "
            f"timestamps={self.timestamps.shape}, timestamp_biases={self.timestamp_biases.shape})"
        )

    @property
    def last_timestamp(self):
        return self.timestamps[-1] if len(self.timestamps) > 0 else float("-inf")

    def append_event(self, event: McapMessage, *, dry_run: bool = False) -> int:
        """Append a new event to context."""
        tokenized_event = episode_tokenizer.tokenize_event(event, is_first=False, is_last=False)

        # append new timestamps
        previous_timestamp = self.timestamps[-1] if len(self.timestamps) > 0 else float("-inf")
        previous_timestamp_bias = self.timestamp_biases[-1] if len(self.timestamp_biases) > 0 else 0
        if event.timestamp < previous_timestamp:
            timestamp_bias = previous_timestamp_bias + episode_tokenizer.encoder.config.timestamp_range
        else:
            timestamp_bias = previous_timestamp_bias

        if dry_run:
            return event.timestamp + int(timestamp_bias)

        # Adjust timestamp
        event.timestamp += int(timestamp_bias)
        # Run callback
        if self.callback is not None:
            self.callback(event)

        self.timestamps = torch.concat([self.timestamps, torch.tensor([event.timestamp], device=self.device)])
        self.timestamp_biases = torch.concat(
            [self.timestamp_biases, torch.tensor([timestamp_bias], device=self.device)]
        )

        # append new event indices
        self.event_indices = torch.concat(
            [self.event_indices, torch.tensor([len(self.sequences)], dtype=torch.long, device=self.device)]
        )
        # append new token ids
        new_token_ids = torch.tensor(tokenized_event["token_ids"], dtype=torch.long, device=self.device)
        logger.debug(
            f"Appending event, {self.sequences.shape[0]} -> {self.sequences.shape[0] + len(new_token_ids)}, images: {self.pixel_values.shape[0]} -> {self.pixel_values.shape[0] + len(tokenized_event['images'])}"
        )
        self.sequences = torch.concat([self.sequences, new_token_ids])
        # append new images
        new_images = tokenized_event["images"]
        self.image_count_per_event = torch.concat(
            [self.image_count_per_event, torch.tensor([len(new_images)], dtype=torch.long, device=self.device)]
        )
        if new_images:
            new_images = [img.to_pil_image() for img in new_images]
            new_pixel_values = processor.image_processor(new_images, return_tensors="pt").pixel_values.to(
                self.device, dtype=torch.bfloat16
            )
            self.pixel_values = torch.concat([self.pixel_values, new_pixel_values])

        self.auto_trim()
        return event.timestamp

    def auto_trim(self):
        """Trim the context to fit into the model's maximum length."""
        while len(self.sequences) > 1024:
            self.pop()

    def pop(self):
        """Pop the first event from context."""
        second_event_index = self.event_indices[1] if len(self.event_indices) > 1 else len(self.sequences)
        logger.debug(
            f"Popping event from context, {self.sequences.shape[0]} -> {self.sequences.shape[0] - second_event_index}, images: {self.pixel_values.shape[0]} -> {self.pixel_values.shape[0] - self.image_count_per_event[0]}"
        )
        self.sequences = self.sequences[second_event_index:]
        self.pixel_values = self.pixel_values[self.image_count_per_event[0] :]
        self.event_indices = self.event_indices[1:] - second_event_index
        self.image_count_per_event = self.image_count_per_event[1:]
        self.timestamps = self.timestamps[1:]
        self.timestamp_biases = self.timestamp_biases[1:]


def generate_single_event(sequences, pixel_values) -> torch.LongTensor:
    """Generate a single event."""

    attention_mask = torch.ones_like(sequences, dtype=torch.bool, device=torch_device)
    logger.debug(f"Generating event from {sequences.shape}, {pixel_values.shape}")
    outputs = model.generate(
        input_ids=sequences,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        eos_token_id=tokenizer.encode("<EVENT_END>")[0],
        max_new_tokens=20,  # event with maximum length must not exceed this value
        use_cache=True,
        # logits_processor=[EventGenerationLogitsProcessor(tokenizer)],
    )[0, sequences.shape[1] :]
    outputs = cast(torch.LongTensor, outputs)
    logger.debug(f"Generated event: {outputs.shape} tokens from {sequences.shape}")
    return outputs


def generate(input_iterator: Iterator[tuple[McapMessage, TokenizedEvent]], writer: OWAMcapWriter):
    context_manager = ContextManager(device=torch_device, callback=writer.write_message)
    context_manager.append_event(next(input_iterator)[0])

    while True:
        try:
            next_event, _ = next(input_iterator)
            # generate events until next_event.timestamp
            while True:
                sequences, pixel_values = context_manager.sequences, context_manager.pixel_values
                # sequences: [seq_len] -> [1, seq_len], pixel_values: [num_images, 3, 448, 448]
                new_sequences = generate_single_event(sequences.unsqueeze(0), pixel_values)
                # If generatd event is not valid event, break
                try:
                    new_event = episode_tokenizer.decode_event(new_sequences.cpu().numpy())
                except EventEncoderError:
                    logger.debug("Generated invalid event. Breaking generation.")
                    break
                # If generated event is after next_event, break
                if next_event.timestamp < context_manager.append_event(new_event, dry_run=True):
                    logger.debug("Generated event is after next_event. Breaking generation.")
                    break

                context_manager.append_event(new_event)
                logger.success(f"Appended event(from generation): {new_event=}, {context_manager=}")

            # After event generation is over, append event from input stream
            context_manager.append_event(next_event)
            logger.info(f"Generation is over, appending next event: {next_event=}, {context_manager=}")
        except StopIteration:
            break


if __name__ == "__main__":
    input_iterator = input_stream()
    with OWAMcapWriter("output.mcap") as writer:
        generate(input_iterator, writer)
