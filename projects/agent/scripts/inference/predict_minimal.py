import sys
from typing import Callable, Iterator, cast

import torch
from loguru import logger
from transformers import AutoModelForImageTextToText, AutoProcessor

from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter
from mcap_owa.highlevel.mcap_msg import McapMessage
from owa.data.encoders import EventEncoderError, HierarchicalEventEncoder
from owa.data.episode_tokenizer import EpisodeTokenizer, TokenizedEvent
from owa.data.processing.resampler import create_resampler

# Setup logging
logger.remove()
logger.add(sys.stderr, level="INFO")

# Configuration
DEVICE = "cuda:0"
MODEL_PATH = "/mnt/harbor/projects/owa/checkpoints/gidm/InternVL3-1B-hf_gidm-100ms-batch128-lr2e5"
DATA_PATH = "/mnt/raid12/datasets/owa_game_dataset_filtered_448/milkclouds00@gmail.com/apex_legends/0805_01.mcap"
MAX_CONTEXT_LENGTH = 1024
MAX_NEW_TOKENS = 20

# Resampling intervals (in nanoseconds)
SCREEN_RESAMPLE_INTERVAL_NS = 50_000_000  # 50ms - standard for screen capture
MOUSE_RESAMPLE_INTERVAL_NS = 50_000_000  # 50ms - standard for mouse events
KEYBOARD_RESAMPLE_INTERVAL_NS = 0  # 0 - pass through all keyboard events

# Load model and tokenizer
model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, device_map=DEVICE, torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
tokenizer = processor.tokenizer

# Setup episode tokenizer
episode_tokenizer = EpisodeTokenizer.from_transformers_model(MODEL_PATH)
episode_tokenizer.prepare_model(tokenizer=tokenizer)
assert isinstance(episode_tokenizer.encoder, HierarchicalEventEncoder), "Only hierarchical event encoder is supported"


def create_input_stream() -> Iterator[tuple[McapMessage, TokenizedEvent]]:
    """Create input stream from MCAP file, starting from middle timestamp."""
    # Setup resamplers for different event types
    resamplers = {
        "screen": create_resampler("screen", min_interval_ns=SCREEN_RESAMPLE_INTERVAL_NS),
        "mouse/raw": create_resampler("mouse/raw", min_interval_ns=MOUSE_RESAMPLE_INTERVAL_NS),
        "keyboard": create_resampler("keyboard", min_interval_ns=KEYBOARD_RESAMPLE_INTERVAL_NS),
    }
    first_timestamp = None
    prev_msg = None
    with OWAMcapReader(DATA_PATH) as reader:
        # Start from middle of the recording
        mid_time = reader.start_time + (reader.end_time - reader.start_time) // 2

        for msg in reader.iter_messages(topics=["screen", "mouse/raw", "keyboard"], start_time=mid_time):
            resampler = resamplers[msg.topic]
            resampler.add_event(msg)

            # Process resampled events
            for resampled_msg in resampler.pop_event():
                if resampled_msg.topic != "screen":
                    logger.warning(
                        f"Ground truth event: {resampled_msg.topic=}, {resampled_msg.timestamp=}, {resampled_msg.decoded}"
                    )
                    continue

                is_first = prev_msg is None
                prev_msg = resampled_msg

                # Bias timestamps to start from 0, which is equivalent to timestamp_bias=0 in ContextManager
                if first_timestamp is None:
                    first_timestamp = resampled_msg.timestamp
                resampled_msg.timestamp -= first_timestamp

                # Tokenize event and resolve image paths
                tokenized_event = episode_tokenizer.tokenize_event(resampled_msg, is_first=is_first, is_last=False)
                tokenized_event["images"] = [img.resolve_relative_path(DATA_PATH) for img in tokenized_event["images"]]

                yield resampled_msg, tokenized_event


class ContextManager:
    """Manages context window for event generation with automatic trimming."""

    def __init__(self, device: str, callback: Callable[[McapMessage], None] | None = None):
        self.device = device
        self.callback = callback

        # Initialize empty tensors for context data
        self.sequences = torch.tensor([], dtype=torch.long, device=device)
        self.pixel_values = torch.tensor([], dtype=torch.bfloat16, device=device)
        self.event_indices = torch.tensor([], dtype=torch.long, device=device)
        self.image_counts = torch.tensor([], dtype=torch.long, device=device)
        self.timestamps = torch.tensor([], dtype=torch.long, device=device)
        self.timestamp_biases = torch.tensor([], dtype=torch.long, device=device)

    def __repr__(self):
        return (
            f"ContextManager(seq_len={len(self.sequences)}, "
            f"images={len(self.pixel_values)}, events={len(self.event_indices)})"
        )

    @property
    def last_timestamp(self) -> float:
        return self.timestamps[-1].item() if len(self.timestamps) > 0 else float("-inf")

    def append_event(self, event: McapMessage, *, dry_run: bool = False) -> int:
        """Append event to context with timestamp bias handling."""
        tokenized_event = episode_tokenizer.tokenize_event(event, is_first=False, is_last=False)

        # Calculate timestamp bias to handle timestamp resets
        prev_timestamp = self.timestamps[-1].item() if len(self.timestamps) > 0 else float("-inf")
        prev_bias = self.timestamp_biases[-1].item() if len(self.timestamp_biases) > 0 else 0

        # Get timestamp range from encoder config
        encoder = cast(HierarchicalEventEncoder, episode_tokenizer.encoder)
        timestamp_range = encoder.config.timestamp_range
        timestamp_bias = prev_bias + timestamp_range if event.timestamp < prev_timestamp else prev_bias

        if dry_run:
            return event.timestamp + int(timestamp_bias)

        # Apply timestamp bias and run callback
        event.timestamp += int(timestamp_bias)
        if self.callback:
            self.callback(event)

        # Update context tensors
        self._append_tensors(event.timestamp, int(timestamp_bias), tokenized_event)
        self._trim_if_needed()
        return event.timestamp

    def _append_tensors(self, timestamp: int, bias: int, tokenized_event: TokenizedEvent):
        """Helper to append tensors to context."""
        # Append timestamps and bias
        self.timestamps = torch.cat([self.timestamps, torch.tensor([timestamp], device=self.device)])
        self.timestamp_biases = torch.cat([self.timestamp_biases, torch.tensor([bias], device=self.device)])

        # Append event start index
        self.event_indices = torch.cat([self.event_indices, torch.tensor([len(self.sequences)], device=self.device)])

        # Append token sequences
        new_tokens = torch.tensor(tokenized_event["token_ids"], dtype=torch.long, device=self.device)
        self.sequences = torch.cat([self.sequences, new_tokens])

        # Append images
        new_images = tokenized_event["images"]
        self.image_counts = torch.cat([self.image_counts, torch.tensor([len(new_images)], device=self.device)])

        if new_images:
            pil_images = [img.to_pil_image() for img in new_images]
            pixel_values = processor.image_processor(pil_images, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device, dtype=torch.bfloat16)
            self.pixel_values = torch.cat([self.pixel_values, pixel_values])

    def _trim_if_needed(self):
        """Trim context to fit model's maximum length."""
        while len(self.sequences) > MAX_CONTEXT_LENGTH:
            self._pop_first_event()

    def _pop_first_event(self):
        """Remove the first event from context."""
        if len(self.event_indices) <= 1:
            return

        # Find where second event starts
        second_event_start = self.event_indices[1].item()
        first_event_images = self.image_counts[0].item()

        # Trim all tensors
        self.sequences = self.sequences[second_event_start:]
        self.pixel_values = self.pixel_values[first_event_images:]
        self.event_indices = self.event_indices[1:] - second_event_start
        self.image_counts = self.image_counts[1:]
        self.timestamps = self.timestamps[1:]
        self.timestamp_biases = self.timestamp_biases[1:]


def generate_single_event(sequences: torch.Tensor, pixel_values: torch.Tensor) -> torch.LongTensor:
    """Generate a single event using the model."""
    attention_mask = torch.ones_like(sequences, dtype=torch.bool, device=DEVICE)

    outputs = model.generate(
        input_ids=sequences,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        eos_token_id=tokenizer.encode("<EVENT_END>")[0],
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True,
    )[0, sequences.shape[1] :]

    return cast(torch.LongTensor, outputs)


def run_generation(input_iterator: Iterator[tuple[McapMessage, TokenizedEvent]], writer: OWAMcapWriter):
    """Main generation loop that processes input events and generates intermediate events."""
    context = ContextManager(device=DEVICE, callback=writer.write_message)

    # Initialize with first event
    first_event, _ = next(input_iterator)
    context.append_event(first_event)

    # Process remaining events
    for next_event, _ in input_iterator:
        # Generate events until we reach the next input event
        while True:
            # Generate new event from current context
            sequences = context.sequences.unsqueeze(0)  # Add batch dimension
            new_tokens = generate_single_event(sequences, context.pixel_values)

            # Try to decode generated event
            try:
                generated_event = episode_tokenizer.decode_event(new_tokens.cpu().numpy())
            except EventEncoderError:
                logger.debug("Generated invalid event, stopping generation")
                break

            # Check if generated event is after next input event
            if next_event.timestamp < context.append_event(generated_event, dry_run=True):
                logger.debug("Generated event is after next input event, stopping generation")
                break

            # Add generated event to context
            context.append_event(generated_event)
            logger.success(
                f"Generated event: {generated_event.topic} at {generated_event.timestamp}, decoded={generated_event.decoded}, {context}"
            )

        # Add the next input event to context
        context.append_event(next_event)
        logger.info(
            f"Added input event: {next_event.topic} at {next_event.timestamp}, decoded={next_event.decoded}, {context}"
        )


if __name__ == "__main__":
    input_iterator = create_input_stream()
    with OWAMcapWriter("output.mcap") as writer:
        run_generation(input_iterator, writer)
