"""
Pure functions for event tokenization.

These functions are stateless and depend only on their inputs.
They require an EventTokenizationContext that contains all necessary dependencies.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterator, List, Literal, TypedDict, Union, overload

import numpy as np
import numpy.typing as npt

from mcap_owa.highlevel import McapMessage
from owa.data.encoders import FactorizedEventEncoder, HierarchicalEventEncoder
from owa.msgs.desktop.screen import ScreenCaptured

from .context import EventTokenizationContext

if TYPE_CHECKING:
    from owa.data.datasets import Dataset


class TokenizedEvent(TypedDict):
    """Result of tokenizing a single event."""

    text: str
    images: List[ScreenCaptured]
    token_ids: List[int]
    total_token_count: int


@overload
def tokenize_event(
    ctx: EventTokenizationContext,
    mcap_msg: McapMessage,
    *,
    return_dict: Literal[True] = True,
) -> TokenizedEvent: ...


@overload
def tokenize_event(
    ctx: EventTokenizationContext,
    mcap_msg: McapMessage,
    *,
    return_dict: Literal[False],
) -> npt.NDArray[np.int64]: ...


def tokenize_event(
    ctx: EventTokenizationContext,
    mcap_msg: McapMessage,
    *,
    return_dict: bool = True,
) -> Union[TokenizedEvent, npt.NDArray[np.int64]]:
    """Tokenize a single McapMessage to token IDs.

    Args:
        ctx: Tokenization context containing encoder, tokenizer, and config
        mcap_msg: The MCAP message to tokenize
        return_dict: If True, return TokenizedEvent dict; if False, return token IDs array

    Returns:
        TokenizedEvent dict or numpy array of token IDs
    """
    # Encode message to text using EventEncoder
    encoded_text, images = ctx.encoder.encode(mcap_msg)

    # Replace fake placeholder with actual image token pattern
    encoded_text = encoded_text.replace(
        ctx.image_config.fake_placeholder,
        ctx.image_config.pattern,
    )

    # Tokenize text to IDs
    token_ids = ctx.tokenizer.encode(encoded_text, add_special_tokens=False, return_tensors="np")[0]

    if return_dict:
        return TokenizedEvent(
            text=encoded_text,
            images=images,
            token_ids=token_ids,
            total_token_count=len(token_ids),
        )
    return token_ids


def decode_event(
    ctx: EventTokenizationContext,
    token_ids: npt.NDArray[np.int64],
) -> McapMessage:
    """Decode token IDs back to a McapMessage.

    Note: This expects image tokens to be excluded (treated as -100 in labels).

    Args:
        ctx: Tokenization context
        token_ids: Token IDs to decode

    Returns:
        Reconstructed McapMessage
    """
    text = ctx.tokenizer.decode(token_ids, skip_special_tokens=False)

    # Verify no image tokens in text
    assert ctx.image_config.token not in text, (
        f"Image token {ctx.image_config.token} found in text. "
        "This method expects image tokens are excluded since they are treated as -100 in labels."
    )

    # Convert empty image pattern back to placeholder
    empty_pattern = f"{ctx.image_config.prefix}{ctx.image_config.suffix}"
    text = text.replace(empty_pattern, ctx.image_config.fake_placeholder)

    return ctx.encoder.decode(text)


def tokenize_episode(
    ctx: EventTokenizationContext,
    mcap_messages: Iterator[McapMessage],
) -> Iterator[TokenizedEvent]:
    """Tokenize a sequence of McapMessages.

    Args:
        ctx: Tokenization context
        mcap_messages: Iterator of MCAP messages

    Yields:
        TokenizedEvent for each message
    """
    for mcap_msg in mcap_messages:
        yield tokenize_event(ctx, mcap_msg)


def decode_episode(
    ctx: EventTokenizationContext,
    input_ids_or_text: Union[List[int], npt.NDArray[np.int64], str],
    *,
    skip_invalid: bool = True,
    adjust_timestamp: bool = True,
) -> Iterator[McapMessage]:
    """Decode token IDs or text back to McapMessages.

    Args:
        ctx: Tokenization context
        input_ids_or_text: Token IDs or tokenized text string
        skip_invalid: If True, skip invalid events instead of raising
        adjust_timestamp: If True, adjust timestamps for modular arithmetic

    Yields:
        Reconstructed McapMessage for each valid event
    """
    if not isinstance(ctx.encoder, (HierarchicalEventEncoder, FactorizedEventEncoder)):
        raise NotImplementedError(
            f"decode_episode is only implemented for HierarchicalEventEncoder and FactorizedEventEncoder, "
            f"got {type(ctx.encoder)}"
        )

    # Convert token IDs to text if needed
    if isinstance(input_ids_or_text, str):
        text = input_ids_or_text
    else:
        text = ctx.tokenizer.decode(input_ids_or_text, skip_special_tokens=False)

    # Parse events between <EVENT_START> and <EVENT_END>
    event_strings = re.findall(r"<EVENT_START>.*?<EVENT_END>", text)

    previous_timestamp = float("-inf")
    timestamp_bias = 0

    for event_string in event_strings:
        try:
            # Convert image pattern back to placeholder
            processed = event_string.replace(
                f"{ctx.image_config.prefix}{ctx.image_config.suffix}",
                ctx.image_config.fake_placeholder,
            )
            event = ctx.encoder.decode(processed)

            if adjust_timestamp:
                timestamp_range = ctx.encoder.config.timestamp_range
                if event.timestamp < previous_timestamp:
                    timestamp_bias += timestamp_range
                event.timestamp += timestamp_bias

            yield event
            previous_timestamp = event.timestamp
        except Exception as e:
            if not skip_invalid:
                raise e


def tokenize_event_dataset(
    ctx: EventTokenizationContext,
    event_dataset: "Dataset",
    map_kwargs: dict = None,
) -> "Dataset":
    """Tokenize an entire event dataset.

    Args:
        ctx: Tokenization context
        event_dataset: Input event dataset (must be EVENT stage)
        map_kwargs: Additional kwargs for dataset.map()

    Returns:
        Tokenized dataset with token_ids and text columns
    """
    from owa.data.datasets import Dataset, DatasetStage

    if map_kwargs is None:
        map_kwargs = {"num_proc": 32}

    if not isinstance(event_dataset, Dataset):
        raise ValueError(f"Expected Dataset from `owa.data.datasets`, got {type(event_dataset)}")

    def process_event(event):
        mcap_message = McapMessage.model_validate_json(event["mcap_message"])
        tokenized = tokenize_event(ctx, mcap_message)

        return {
            "episode_path": event["episode_path"],
            "topic": event["topic"],
            "timestamp_ns": event["timestamp_ns"],
            "text": tokenized["text"],
            "images": [image.model_dump_json() for image in tokenized["images"]],
            "token_ids": tokenized["token_ids"],
            "total_token_count": tokenized["total_token_count"],
        }

    tokenized_dataset = event_dataset.map(
        process_event,
        desc="Tokenizing event dataset",
        remove_columns=event_dataset.column_names,
        **map_kwargs,
    )

    # Convert back to OWA Dataset
    tokenized_dataset = Dataset.from_hf_dataset(
        tokenized_dataset,
        owa_config=event_dataset.owa_config,
    )
    tokenized_dataset.owa_config.stage = DatasetStage.TOKENIZED

    return tokenized_dataset
