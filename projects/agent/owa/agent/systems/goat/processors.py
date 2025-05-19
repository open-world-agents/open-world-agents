import warnings
from typing import Any

import torch
from pydantic import BaseModel
from transformers import GPT2TokenizerFast
from transformers.models.smolvlm import SmolVLMProcessor

from owa.agent.core.perception import Perception, PerceptionSpecDict, apply_spec
from owa.env.gst.msg import ScreenEmitted

from .perception_spec import PERCEPTION_SPEC_DICT
from .utils import EventProcessor

CHAT_INSTRUCTION = """
You are playing Super Hexagon, a fast-paced game that requires precise control and timing.
Given past event trajectory, predict the future sequence of event.
"""


class SmolVLMInput(BaseModel):
    images: list[Any]
    messages: list[dict]


def perception_to_conversation(
    perception_history: Perception,
    current_perception: Perception,
    *,
    now: int,
    is_training: bool,
    spec: PerceptionSpecDict | None = None,
    event_processor: EventProcessor | None = None,
) -> tuple[Perception, SmolVLMInput | None]:
    """For events later than 'now', it's considered as future events('label')."""

    if spec is None:
        spec = PERCEPTION_SPEC_DICT
        warnings.warn("PerceptionSpecDict is not provided. Using default PERCEPTION_SPEC_DICT.")

    if event_processor is None:
        event_processor = EventProcessor()
        warnings.warn("EventProcessor is not provided. Using default EventProcessor.")

    perception_history += current_perception
    perception_history, info = apply_spec(perception_history, now=now, spec=spec)
    if perception_history:
        event_history = sorted(
            perception_history["inputs/keyboard"]
            + perception_history["inputs/mouse"]
            + perception_history["inputs/screen"],
            key=lambda x: x.timestamp,
        )
        tokenized_history = "".join(event_processor.tokenize(event_history, now=now))

        event_label = sorted(
            perception_history["outputs/keyboard"] + perception_history["outputs/mouse"], key=lambda x: x.timestamp
        )
        tokenized_label = "".join(event_processor.tokenize(event_label, now=now))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CHAT_INSTRUCTION + tokenized_history},
                ],
            }
        ]
        if is_training:
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": tokenized_label},
                    ],
                }
            )
        conversation = SmolVLMInput(images=perception_history["inputs/screen"], messages=messages)
    else:
        conversation = None
    return perception_history, conversation


def lazy_load_images(conversation):
    images = []
    for event in conversation.images:
        msg: ScreenEmitted = event.msg
        images.append(msg.to_pil_image())
    conversation.images = images
    return conversation


def apply_processor(examples: SmolVLMInput | list[SmolVLMInput], *, processor: SmolVLMProcessor, is_training: bool):
    if not isinstance(examples, list):
        examples = [examples]

    texts = []
    images = []
    for example in examples:
        if is_training:
            # strip is to remove appended "\n"
            text = processor.apply_chat_template(example.messages, tokenize=False).strip()
        else:
            # NOTE: single space " " must be appended to match the training time tokenization.
            text = processor.apply_chat_template(example.messages, tokenize=False, add_generation_prompt=True) + " "
        texts.append(text)
        images.append(example.images)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    # TODO: return_assistant_tokens_mask. https://github.com/huggingface/transformers/issues/36713
    if is_training:
        tokenizer: GPT2TokenizerFast = processor.tokenizer

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels: torch.IntTensor = batch["input_ids"].clone()

        # First, mask all tokens (set to -100)
        labels.fill_(-100)

        # Pattern for start of assistant's response
        start_pattern = tokenizer("<end_of_utterance>\nAssistant: ")["input_ids"]
        # End pattern is just the <end_of_utterance> token
        end_token = tokenizer("<end_of_utterance>")["input_ids"][0]

        # Process each sequence in the batch
        for i in range(labels.size(0)):
            input_ids = batch["input_ids"][i].tolist()

            # Find the start pattern
            for j in range(len(input_ids) - len(start_pattern)):
                if input_ids[j : j + len(start_pattern)] == start_pattern:
                    start_idx = j + len(start_pattern)

                    # Find the next end token after the start pattern
                    for k in range(start_idx, len(input_ids)):
                        if input_ids[k] == end_token:
                            end_idx = k + 1  # include <end_of_utterance> token

                            # Unmask the assistant's response (keep these tokens in the loss computation)
                            labels[i, start_idx:end_idx] = batch["input_ids"][i, start_idx:end_idx]
                            break
                    break

        batch["labels"] = labels
    return batch
