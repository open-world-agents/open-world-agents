import re
from dataclasses import dataclass
from typing import Iterator, Literal, Sequence, TypedDict, overload

import numpy as np
import numpy.typing as npt
from transformers.tokenization_utils import PreTrainedTokenizer

from mcap_owa.highlevel import McapMessage
from owa.data.encoders import HierarchicalEventEncoder, create_encoder
from owa.msgs.desktop.screen import ScreenCaptured

from .datasets import Dataset, DatasetStage


@dataclass
class EpisodeTokenizerConfig:
    """Configuration for EpisodeTokenizer."""

    encoder_type: str = "hierarchical"
    # Internal placeholder token used by encoders (not a real token, not in vocab)
    fake_image_placeholder: str = "<fake_image_placeholder>"
    # Real image token that will be repeated in the final tokenized sequence
    image_token_prefix: str = "<fake_token_around_image><global-img>"
    image_token: str = "<image>"
    image_token_length: int = 64
    image_token_suffix: str = "<fake_token_around_image>"
    episode_start_token: str = "<EPISODE_START>"
    episode_end_token: str = "<EPISODE_END>"


class TokenizedEvent(TypedDict):
    text: str
    images: list[ScreenCaptured]
    token_ids: list[int]
    total_token_count: int


class EpisodeTokenizer:
    def __init__(self, config: EpisodeTokenizerConfig = EpisodeTokenizerConfig(), **kwargs):
        self.config = EpisodeTokenizerConfig(**(config.__dict__ | kwargs))
        self.encoder = create_encoder(
            self.config.encoder_type,
            fake_image_placeholder=self.config.fake_image_placeholder,
        )
        self.is_prepared = False

    def get_vocab(self) -> set[str]:
        # NOTE: fake_image_placeholder is NOT included as it's not a real token
        # TODO: image_token_prefix or similar things can be composed of multiple tokens, so we need to parse them
        return self.encoder.get_vocab() | {
            self.config.image_token,
            self.config.image_token_prefix,
            self.config.image_token_suffix,
            self.config.episode_start_token,
            self.config.episode_end_token,
        }

    def prepare_model(self, *, tokenizer: PreTrainedTokenizer, model=None):
        tokenizer.add_tokens(sorted(self.get_vocab()))  # NOTE: set is unordered in python
        if model is not None:
            model.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer
        self.is_prepared = True

    def tokenize_event(
        self,
        mcap_msg: McapMessage,
        *,
        is_first: bool = False,
        is_last: bool = False,
    ) -> TokenizedEvent:
        if not self.is_prepared:
            raise RuntimeError("EpisodeTokenizer must be prepared by `prepare_model` before tokenizing")

        encoded_text, images = self.encoder.encode(mcap_msg)

        # Add episode start/end tokens if needed
        prefix = suffix = ""
        if is_first:
            prefix = self.config.episode_start_token
        if is_last:
            suffix = self.config.episode_end_token
        encoded_text = f"{prefix}{encoded_text}{suffix}"

        # Replace fake image placeholder with prefix + repeated real image tokens + suffix
        # EventEncoder outputs fake_image_placeholder, we convert to real image tokens
        replacement = f"{self.config.image_token_prefix}{self.config.image_token * self.config.image_token_length}{self.config.image_token_suffix}"
        encoded_text = encoded_text.replace(self.config.fake_image_placeholder, replacement)
        token_ids = self.tokenizer.encode(encoded_text, add_special_tokens=False)

        return TokenizedEvent(
            text=encoded_text,
            images=images,
            token_ids=token_ids,
            total_token_count=len(token_ids),
        )

    def decode_event(self, tokenized_event: dict) -> McapMessage:
        text = tokenized_event["text"]
        # Convert repeated image token sequences back to fake_image_placeholder
        # Pattern: prefix + (image_token * image_token_length) + suffix -> fake_image_placeholder
        assert self.config.image_token not in text, (
            f"Image token {self.config.image_token} found in text, note that EpisodeTokenizer.decode is intended to be called on evaluation only. (since image tokens are replaced as -100 they're skipped on eval time)"
        )
        repeated_image_pattern = f"{self.config.image_token_prefix}{self.config.image_token_suffix}"
        text = text.replace(repeated_image_pattern, self.config.fake_image_placeholder)

        return self.encoder.decode(text, tokenized_event["images"])

    def tokenize_episode(self, mcap_messages: list[McapMessage]) -> list[TokenizedEvent]:
        return [self.tokenize_event(mcap_msg) for mcap_msg in mcap_messages]

    def decode_episode(
        self,
        input_ids_or_text: list[int] | npt.NDArray[np.int64] | str,
        *,
        skip_invalid: bool = True,
        adjust_timestamp: bool = True,
    ) -> Iterator[McapMessage]:
        """Decode token IDs or tokenized text back to the original McapMessage format."""
        if not isinstance(self.encoder, HierarchicalEventEncoder):
            raise NotImplementedError(
                "EpisodeTokenizer.decode_episode is only implemented for HierarchicalEventEncoder"
            )

        # Convert token IDs back to text (if input is token IDs)
        if isinstance(input_ids_or_text, str):
            text = input_ids_or_text
        else:
            if not self.is_prepared:
                raise RuntimeError("EpisodeTokenizer must be prepared by `prepare_model` before decoding")
            # Input is token IDs
            text = self.tokenizer.decode(input_ids_or_text, skip_special_tokens=False)

        # Remove episode start/end tokens if present
        if text.startswith(self.config.episode_start_token):
            text = text[len(self.config.episode_start_token) :]
        if text.endswith(self.config.episode_end_token):
            text = text[: -len(self.config.episode_end_token)]

        # Parse all events between <EVENT_START> and <EVENT_END> tokens
        event_strings = re.findall(r"<EVENT_START>.*?<EVENT_END>", text)

        previous_timestamp = float("-inf")
        timestamp_bias = 0
        for event_string in event_strings:
            try:
                event = self.decode_event({"text": event_string, "images": None})
                if event.timestamp < previous_timestamp:
                    timestamp_bias += self.encoder.config.timestamp_range
                if adjust_timestamp:
                    event.timestamp += timestamp_bias

                yield event
                previous_timestamp = event.timestamp
            except Exception as e:
                if not skip_invalid:
                    raise e

    def tokenize_event_dataset(self, event_dataset: Dataset, map_kwargs: dict = {"num_proc": 32}) -> Dataset:
        # Check if the input is a Dataset
        if not isinstance(event_dataset, Dataset):
            raise ValueError(f"Expected Dataset from `owa.data.datasets`, got {type(event_dataset)}")

        # Tokenize each event in the dataset
        def process_event(event, idx):
            is_first = is_last = False
            # Add episode start token
            if idx == 0 or (idx > 0 and event["episode_path"] != event_dataset[idx - 1]["episode_path"]):
                is_first = True
            # Add episode end token
            if idx < len(event_dataset) - 1 and event["episode_path"] != event_dataset[idx + 1]["episode_path"]:
                is_last = True

            episode_path = event["episode_path"]
            mcap_message = McapMessage.model_validate_json(event["mcap_message"])
            tokenized_event = self.tokenize_event(mcap_message, is_first=is_first, is_last=is_last)

            return {
                "episode_path": episode_path,
                "text": tokenized_event["text"],
                "token_ids": tokenized_event["token_ids"],
                "images": [image.model_dump_json() for image in tokenized_event["images"]],
                "total_token_count": tokenized_event["total_token_count"],
            }

        # Tokenize the dataset
        tokenized_dataset = event_dataset.map(
            process_event,
            with_indices=True,
            desc="Tokenizing event dataset",
            remove_columns=event_dataset.column_names,
            **map_kwargs,
        )

        # Switch back to OWA Dataset from HF Dataset
        tokenized_dataset = Dataset.from_hf_dataset(tokenized_dataset, owa_config=event_dataset.owa_config)
        tokenized_dataset.owa_config.stage = DatasetStage.TOKENIZED

        return tokenized_dataset


# Inefficient pscan impl
def pscan(dataset: Dataset, round_n: int = 0, map_kwargs: dict = {"num_proc": 32}):
    if len(dataset) - 1 <= (1 << round_n):
        return dataset

    def fn(example, idx):
        if idx & (1 << round_n):
            example["cumulative_token_count"] += dataset[idx - (1 << round_n)]["cumulative_token_count"]
        return example

    dataset = dataset.map(fn, with_indices=True, desc=f"PScan round {round_n}", **map_kwargs)
    dataset = pscan(dataset, round_n + 1, map_kwargs)
    return dataset
