import re
from dataclasses import dataclass
from typing import Iterator, Literal, TypedDict, overload

import numpy as np
import numpy.typing as npt
from transformers.tokenization_utils import PreTrainedTokenizer

from mcap_owa.highlevel import McapMessage
from owa.data.encoders import HierarchicalEventEncoder, create_encoder
from owa.msgs.desktop.screen import ScreenCaptured

from .collator import ModelType, detect_model_type
from .datasets import Dataset, DatasetStage


@dataclass
class EpisodeTokenizerConfig:
    """Configuration for EpisodeTokenizer."""

    # Real image token pattern: f"{image_token_prefix}{image_token * image_token_length}{image_token_suffix}"
    image_token_prefix: str
    image_token: str
    image_token_length: int
    image_token_suffix: str

    encoder_type: str = "hierarchical"
    # Internal placeholder token used by encoders (not in vocab)
    fake_image_placeholder: str = "<fake_image_placeholder>"
    episode_start_token: str = "<EPISODE_START>"
    episode_end_token: str = "<EPISODE_END>"


class TokenizedEvent(TypedDict):
    text: str
    images: list[ScreenCaptured]
    token_ids: list[int]
    total_token_count: int


class EpisodeTokenizer:
    def __init__(self, config: EpisodeTokenizerConfig, **kwargs):
        self.config = EpisodeTokenizerConfig(**(config.__dict__ | kwargs))
        self.encoder = create_encoder(
            self.config.encoder_type,
            fake_image_placeholder=self.config.fake_image_placeholder,
        )
        self.is_prepared = False

    @classmethod
    def from_transformers_model(cls, model_name_or_path: str, **kwargs):
        model_type = detect_model_type(model_name_or_path)
        if model_type == ModelType.INTERNVL:
            # InternVL3 configuration
            episode_tokenizer_config = EpisodeTokenizerConfig(
                encoder_type="hierarchical",
                fake_image_placeholder="<fake_image_placeholder>",
                image_token_prefix="<img>",
                image_token="<IMG_CONTEXT>",
                image_token_length=256,
                image_token_suffix="</img>",
                episode_start_token="<EPISODE_START>",
                episode_end_token="<EPISODE_END>",
            )
        else:
            # SmolVLM2 and other models configuration
            episode_tokenizer_config = EpisodeTokenizerConfig(
                encoder_type="hierarchical",
                fake_image_placeholder="<fake_image_placeholder>",
                image_token_prefix="<fake_token_around_image><global-img>",
                image_token="<image>",
                image_token_length=64,
                image_token_suffix="<fake_token_around_image>",
                episode_start_token="<EPISODE_START>",
                episode_end_token="<EPISODE_END>",
            )
        return cls(episode_tokenizer_config, **kwargs)

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

    @overload
    def tokenize_event(
        self,
        mcap_msg: McapMessage,
        *,
        is_first: bool = False,
        is_last: bool = False,
        return_dict: Literal[True] = True,
    ) -> TokenizedEvent: ...

    @overload
    def tokenize_event(
        self,
        mcap_msg: McapMessage,
        *,
        is_first: bool = False,
        is_last: bool = False,
        return_dict: Literal[False],
    ) -> npt.NDArray[np.int64]: ...

    def tokenize_event(
        self,
        mcap_msg: McapMessage,
        *,
        is_first: bool = False,
        is_last: bool = False,
        return_dict: bool = True,
    ) -> TokenizedEvent | npt.NDArray[np.int64]:
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
        token_ids = self.tokenizer.encode(encoded_text, add_special_tokens=False, return_tensors="np")[0]

        if return_dict:
            return TokenizedEvent(
                text=encoded_text,
                images=images,
                token_ids=token_ids,
                total_token_count=len(token_ids),
            )
        else:
            return token_ids

    def decode_event(self, token_ids: npt.NDArray[np.int64]) -> McapMessage:
        text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        # Convert repeated image token sequences back to fake_image_placeholder
        # Pattern: prefix + (image_token * image_token_length) + suffix -> fake_image_placeholder
        assert self.config.image_token not in text, (
            f"Image token {self.config.image_token} found in text, note that this method expects image tokens are excluded since they are treated as -100 in labels."
        )
        repeated_image_pattern = f"{self.config.image_token_prefix}{self.config.image_token_suffix}"
        text = text.replace(repeated_image_pattern, self.config.fake_image_placeholder)

        return self.encoder.decode(text)

    def tokenize_episode(
        self,
        mcap_messages: Iterator[McapMessage],
        *,
        append_episode_start_token: bool = False,
        append_episode_end_token: bool = False,
    ) -> Iterator[TokenizedEvent]:
        iterator = iter(mcap_messages)
        prev_msg = None
        current_msg = next(iterator, None)

        while current_msg is not None:
            next_msg = next(iterator, None)
            is_first = prev_msg is None
            is_last = next_msg is None

            yield self.tokenize_event(
                current_msg,
                is_first=is_first and append_episode_start_token,
                is_last=is_last and append_episode_end_token,
            )

            prev_msg = current_msg
            current_msg = next_msg

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

        # Initialize previous timestamp and timestamp bias
        previous_timestamp = float("-inf")
        timestamp_bias = 0
        for event_string in event_strings:
            try:
                event = self.decode_event({"text": event_string, "images": None})
                # Since HierarchicalEventEncoder uses modular arithmetic for timestamp encoding,
                # we need to adjust the timestamp if it's smaller than the previous one
                if event.timestamp < previous_timestamp:
                    timestamp_bias += self.encoder.config.timestamp_range
                # Adjust timestamp with bias
                if adjust_timestamp:
                    event.timestamp += timestamp_bias

                yield event
                # Update previous timestamp
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
