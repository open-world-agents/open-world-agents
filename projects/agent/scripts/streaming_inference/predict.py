import copy
from typing import Iterator

import torch
from loguru import logger
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    PreTrainedTokenizer,
)
from transformers.cache_utils import Cache, StaticCache
from transformers.generation import LogitsProcessor

from mcap_owa.highlevel import OWAMcapReader
from mcap_owa.highlevel.mcap_msg import McapMessage
from owa.data.datasets import load_from_disk
from owa.data.encoders import EventEncoderError
from owa.data.episode_tokenizer import EpisodeTokenizer, TokenizedEvent
from owa.data.processing.resampler import create_resampler

torch_device = "cuda:0"

"""
@dataclass
class GenerateDecoderOnlyOutput:
    sequences: torch.LongTensor
    scores: Optional[tuple[torch.FloatTensor]] = None
    logits: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[tuple[tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[tuple[tuple[tuple[torch.FloatTensor]]]] = None
"""

pretrained_model_name_or_path = "/mnt/harbor/projects/owa/checkpoints/gidm/InternVL3-1B-hf_gidm-100ms-batch128-lr2e5"
model = AutoModelForImageTextToText.from_pretrained(
    pretrained_model_name_or_path, device_map=torch_device, torch_dtype=torch.bfloat16
)
processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path)
tokenizer = processor.tokenizer


class EventGenerationLogitsProcessor(LogitsProcessor):
    """
    Generate structured events. <EVENT_START>(TYPE OF EVENT)(TIMESTAMP OF EVENT)(EVENT DATA)<EVENT_END>

    TODO: this class is not complete. Finish this.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.event_start_tokens = tokenizer.encode("<EVENT_START>", add_special_tokens=False)
        self.event_end_tokens = tokenizer.encode("<EVENT_END>", add_special_tokens=False)
        self.event_type_token_ids = [
            tokenizer.encode("<KEYBOARD>", add_special_tokens=False)[0],
            tokenizer.encode("<MOUSE>", add_special_tokens=False)[0],
        ]
        self.filter_value = -float("inf")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids[0, -len(self.event_end_tokens) :].tolist() == self.event_end_tokens:
            scores[:, :] = self.filter_value
            scores[:, self.event_start_tokens[0]] = 0  # allow <EVENT_START>
        elif input_ids[0, -len(self.event_start_tokens) :].tolist() == self.event_start_tokens:
            mask = torch.ones_like(scores, dtype=torch.bool)
            mask[:, self.event_type_token_ids] = False
            scores = scores.masked_fill(mask, self.filter_value)  # mask out event type tokens
        return scores


def test_generate(model, processor, tokenizer):
    dataset = load_from_disk("/mnt/harbor/projects/owa/data/gidm-100ms-fsl-internvl3")
    dataset.auto_set_transform(
        stage="fsl", load_images=True, image_processor=processor.image_processor, pad_token_id=tokenizer.pad_token_id
    )
    dataset = dataset["test"]
    sample = dataset[0]

    texts: str = sample["texts"]
    event_start_positions = []
    event_count = 0
    search_start = 0
    while True:
        pos = texts.find("<EVENT_START>", search_start)
        if pos == -1:
            break
        event_start_positions.append(pos)
        event_count += 1
        search_start = pos + 1  # or pos + len("<EVENT_START>") if you don't want overlapping

    event_count_to_use = 9
    print(event_start_positions)
    print(f"Using {event_count_to_use} events out of {event_count}")  # 8 out of 21

    texts = sample["texts"][: event_start_positions[event_count_to_use - 1]]
    image_event_count = texts.count("<img>")
    input_ids: torch.LongTensor = tokenizer.encode(texts, add_special_tokens=False, return_tensors="pt").to(
        model.device
    )  # [1, 1108]
    pixel_values: torch.FloatTensor = sample["images"][:image_event_count]  # [4, 3, 448, 448]

    input_ids = input_ids.to(model.device)
    pixel_values = pixel_values.to(model.device, dtype=torch.bfloat16)
    print(
        input_ids.shape, input_ids.device, input_ids.dtype, pixel_values.shape, pixel_values.device, pixel_values.dtype
    )

    past_key_values = None
    outputs = model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        past_key_values=past_key_values,
        eos_token_id=tokenizer.encode("<EVENT_END>")[0],
        max_new_tokens=20,
        use_cache=True,
        return_dict_in_generate=True,
    )
    sequences = outputs.sequences
    past_key_values = outputs.past_key_values
    gt = tokenizer.decode(sample["input_ids"][input_ids.shape[1] :][:20])
    decoded_output = tokenizer.decode(sequences[0, input_ids.shape[1] :], skip_special_tokens=False)
    print(gt)  # <EVENT_START><MOUSE><15><9><1><0><1><0><19><0><6><9><9><0><0><0><EVENT_END><EVENT_START><SCREEN><15>
    print(decoded_output)  # <EVENT_START><MOUSE><15><9><1><0><1><0><19><0><7><7><5><0><0><0><EVENT_END>


# test_generate(model, processor, tokenizer)
# exit()


episode_tokenizer = EpisodeTokenizer.from_transformers_model(pretrained_model_name_or_path)
episode_tokenizer.prepare_model(tokenizer=tokenizer)


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
                    logger.info(f"Ground truth event: {mcap_msg.topic=}, {mcap_msg.timestamp=}, {mcap_msg.decoded}")
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


def prepare_initial_inputs(input_iterator) -> tuple[torch.LongTensor, torch.FloatTensor, Cache]:
    """Prepare initial inputs for generation."""
    sequences = torch.tensor([], dtype=torch.long, device=torch_device)
    pixel_values = torch.tensor([], dtype=torch.bfloat16, device=torch_device)
    # BUG: DynamicCache doesn't support cache_position. WHAT?!
    past_key_values = StaticCache(model.config, max_cache_len=4096)  # TODO: implement custom cache

    for mcap_msg, tokenized_event in input_iterator:
        # append new token ids
        new_token_ids = torch.tensor(tokenized_event["token_ids"], dtype=torch.long, device=torch_device)
        # TODO: check here
        # if len(sequences) + len(new_token_ids) > 1024:
        if len(sequences) > 0:
            break
        sequences = torch.concat([sequences, new_token_ids])

        # append new images
        new_images = tokenized_event["images"]
        new_images = [img.to_pil_image() for img in new_images]
        new_pixel_values = processor.image_processor(new_images, return_tensors="pt").pixel_values.to(
            torch_device, dtype=torch.bfloat16
        )
        pixel_values = torch.concat([pixel_values, new_pixel_values])

    sequences = sequences.unsqueeze(0)  # [1, seq_len]
    logger.info(f"Prepared initial input: {sequences.shape=}, {pixel_values.shape=}")
    return sequences, pixel_values, past_key_values


# BUG: when we pass prefilled past_key_values to generate, it crashes
#   File "/mnt/home/claude/GitHub/transformers/src/transformers/generation/utils.py", line 475, in _cache_dependant_input_preparation
#     or (cache_position[-1] >= input_ids.shape[1])  # Exception 3
#         ~~~~~~~~~~~~~~^^^^
# IndexError: index -1 is out of bounds for dimension 0 with size 0
def prefill(sequences, pixel_values, past_key_values: Cache):
    # Case when already prefilled
    logger.info(f"Prefilling cache: {sequences.shape=}, {past_key_values.get_seq_length()=}, {type(past_key_values)=}")
    if past_key_values.get_seq_length() >= sequences.shape[1]:
        return past_key_values
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    sequences = sequences[:, past_seen_tokens:]
    cache_position = torch.arange(past_seen_tokens, past_seen_tokens + sequences.shape[1], device=sequences.device)
    # BUG: DynamicCache doesn't support cache_position. WHAT?!
    # prefill. Reference: https://huggingface.co/docs/transformers/main/kv_cache#prefill-a-cache-prefix-caching
    with torch.no_grad():
        past_key_values = model(
            input_ids=sequences,
            pixel_values=pixel_values,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True,
        ).past_key_values
    logger.info(f"Prefilled cache: {sequences.shape=}, {past_key_values.get_seq_length()=}, {type(past_key_values)=}")
    return past_key_values


def generate_single_event(sequences, pixel_values, past_key_values):
    """Generate a single event. Constrained decoding prevents (1) generation of screen event (2) generation of invalid events."""

    attention_mask = torch.ones_like(sequences, dtype=torch.bool, device=torch_device)
    outputs = model.generate(
        input_ids=sequences,
        pixel_values=pixel_values,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        eos_token_id=tokenizer.encode("<EVENT_END>")[0],
        max_new_tokens=20,
        use_cache=True,
        return_dict_in_generate=True,
        # logits_processor=[EventGenerationLogitsProcessor(tokenizer)],
    )
    logger.info(
        f"Generated event: {sequences.shape} -> {outputs.sequences.shape}, {past_key_values[0][0].shape} -> {outputs.past_key_values[0][0].shape} from {pixel_values.shape=}"
    )
    sequences = outputs.sequences
    past_key_values = outputs.past_key_values
    return sequences, past_key_values


def generate():
    input_iterator = input_stream()
    last_token_count = 0
    last_event_timestamp = float("-inf")
    timestamp_bias = 0

    sequences, pixel_values, past_key_values = prepare_initial_inputs(input_iterator)
    last_token_count = sequences.shape[1]

    while True:
        try:
            next_event, next_event_tokenized = next(input_iterator)
            # generate events until next_event.timestamp
            while True:
                # past_key_values = prefill(sequences, pixel_values, past_key_values)
                past_past_key_values = copy.deepcopy(past_key_values)
                new_sequences, new_past_key_values = generate_single_event(sequences, pixel_values, past_key_values)
                # If generatd event is not valid event, break
                try:
                    new_event = episode_tokenizer.decode_event(new_sequences[0, last_token_count:])
                except EventEncoderError:
                    # If generated event is invalid, revert past_key_values
                    past_key_values = past_past_key_values
                    logger.info("Generated invalid event. Reverting past_key_values.")
                    break
                # If generated event is after next_event, break
                if next_event.timestamp + timestamp_bias < new_event.timestamp:
                    # If generated event is after next_event, revert past_key_values
                    past_key_values = past_past_key_values
                    logger.info("Generated event is after next_event. Reverting past_key_values.")
                    break

                # adjust timestamp
                if new_event.timestamp < last_event_timestamp:
                    timestamp_bias += episode_tokenizer.encoder.config.timestamp_range
                new_event.timestamp += timestamp_bias

                # ===== update state =====
                last_token_count = new_sequences.shape[1]
                last_event_timestamp = new_event.timestamp
                sequences = new_sequences
                past_key_values = new_past_key_values
                logger.info(f"Appended event(from generation): {new_event}")
                # ========================

            # adjust timestamp
            if next_event.timestamp < last_event_timestamp:
                timestamp_bias += episode_tokenizer.encoder.config.timestamp_range
            next_event.timestamp += timestamp_bias

            # ===== update state =====
            # append new token ids
            new_token_ids = torch.tensor(
                next_event_tokenized["token_ids"], dtype=torch.long, device=torch_device
            ).unsqueeze(0)  # [1, new_token_count]
            sequences = torch.concat([sequences, new_token_ids], dim=1)

            # append new images
            new_images = next_event_tokenized["images"]
            new_images = [img.to_pil_image() for img in new_images]
            new_pixel_values = processor.image_processor(new_images, return_tensors="pt").pixel_values.to(
                torch_device, dtype=torch.bfloat16
            )
            pixel_values = torch.concat([pixel_values, new_pixel_values], dim=0)

            last_token_count = sequences.shape[1]
            last_event_timestamp = next_event.timestamp
            logger.info(f"Appended event(from input stream): {next_event}")
            # ========================
        except StopIteration:
            break


if __name__ == "__main__":
    generate()
