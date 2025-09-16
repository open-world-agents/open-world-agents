import re
from dataclasses import dataclass
from typing import Iterator, Literal, TypedDict, overload

import numpy as np
import numpy.typing as npt
from loguru import logger
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from mcap_owa.highlevel import McapMessage
from owa.data.encoders import FactorizedEventEncoder, HierarchicalEventEncoder, create_encoder
from owa.msgs.desktop.screen import ScreenCaptured

from .collator import ModelType, detect_model_type
from .datasets import Dataset, DatasetStage
from .semantic_init import apply_semantic_initialization


@dataclass
class EpisodeTokenizerConfig:
    """Configuration for EpisodeTokenizer."""

    # Real image token pattern: f"{image_token_prefix}{image_token * image_token_length}{image_token_suffix}"
    image_token_prefix: str
    image_token: str
    image_token_length: int
    image_token_suffix: str

    encoder_type: str = "factorized"
    # Internal placeholder token used by encoders (not in vocab)
    fake_image_placeholder: str = "<fake_image_placeholder>"


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
    def from_transformers(cls, model_name_or_path: str, encoder_type: str | None = None, **kwargs):
        model_type = detect_model_type(model_name_or_path)

        # Get base configuration for the model type
        if model_type == ModelType.INTERNVL:
            # InternVL3 configuration
            base_config = EpisodeTokenizerConfig(
                encoder_type="hierarchical",  # Will be overridden
                fake_image_placeholder="<fake_image_placeholder>",
                image_token_prefix="<img>",
                image_token="<IMG_CONTEXT>",
                image_token_length=256,
                image_token_suffix="</img>",
            )
        else:
            # SmolVLM2 and other models configuration
            base_config = EpisodeTokenizerConfig(
                encoder_type="hierarchical",  # Will be overridden
                fake_image_placeholder="<fake_image_placeholder>",
                image_token_prefix="<fake_token_around_image><global-img>",
                image_token="<image>",
                image_token_length=64,
                image_token_suffix="<fake_token_around_image>",
            )

        # Handle encoder type detection/validation
        if encoder_type is None:
            # Try to detect encoder type from model's tokenizer vocab
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            tokenizer_vocab = set(tokenizer.get_vocab().keys())
            detected_encoder_type = cls.detect_encoder_type(tokenizer_vocab, base_config)

            if detected_encoder_type is None:
                raise ValueError(
                    f"Cannot auto-detect encoder type for model '{model_name_or_path}'. "
                    f"The tokenizer vocabulary appears to be unexpanded (original model vocab). "
                    f"Please specify encoder_type explicitly: 'hierarchical' or 'factorized'."
                )

            encoder_type = detected_encoder_type
            logger.info(f"Auto-detected encoder type: {encoder_type}")
        else:
            logger.info(f"Using explicit encoder type: {encoder_type}")

        # Apply encoder type and any additional overrides
        final_config = EpisodeTokenizerConfig(**{**base_config.__dict__, "encoder_type": encoder_type, **kwargs})
        return cls(final_config)

    @classmethod
    def detect_encoder_type(cls, tokenizer_vocab: set[str], base_config: EpisodeTokenizerConfig) -> str | None:
        """Detect encoder type from tokenizer vocabulary."""
        # Test hierarchical encoder vocab
        hierarchical_config = EpisodeTokenizerConfig(**{**base_config.__dict__, "encoder_type": "hierarchical"})
        hierarchical_tokenizer = cls(hierarchical_config)
        hierarchical_vocab = hierarchical_tokenizer.get_vocab()

        # Test factorized encoder vocab
        factorized_config = EpisodeTokenizerConfig(**{**base_config.__dict__, "encoder_type": "factorized"})
        factorized_tokenizer = cls(factorized_config)
        factorized_vocab = factorized_tokenizer.get_vocab()

        # Check if tokenizer vocab contains all tokens from either encoder
        hierarchical_match = hierarchical_vocab.issubset(tokenizer_vocab)
        factorized_match = factorized_vocab.issubset(tokenizer_vocab)

        # Determine encoder type based on vocab match
        if hierarchical_match and factorized_match:
            # Both match - this shouldn't happen. Raise error
            raise ValueError(
                "Tokenizer vocab matches both hierarchical and factorized encoders. "
                "Expected only one match. Please report this issue."
            )
        elif hierarchical_match:
            return "hierarchical"
        elif factorized_match:
            return "factorized"
        elif len(tokenizer_vocab.intersection(hierarchical_vocab)) > len(base_config.__dict__):
            # Vocab has some encoder tokens but not complete - likely expanded but corrupted
            raise ValueError(
                f"Tokenizer vocab appears to be expanded (has {len(tokenizer_vocab.intersection(hierarchical_vocab))} "
                f"encoder tokens) but doesn't match any known encoder type. "
                f"Expected hierarchical ({len(hierarchical_vocab)} tokens) or factorized ({len(factorized_vocab)} tokens)."
            )
        else:
            # Vocab is not expanded
            return None

    def get_vocab(self) -> set[str]:
        # NOTE: fake_image_placeholder is NOT included as it's not a real token
        # TODO: image_token_prefix or similar things can be composed of multiple tokens, so we need to parse them
        return self.encoder.get_vocab() | {
            self.config.image_token,
            self.config.image_token_prefix,
            self.config.image_token_suffix,
        }

    def prepare_model(self, *, tokenizer: PreTrainedTokenizer, model=None, apply_semantic_init: bool = True):
        special_tokens = self.get_vocab()
        vocab = tokenizer.get_vocab()
        if all(tok in vocab for tok in special_tokens):
            logger.warning("Model already has expanded vocab, skipping token addition")
            self.tokenizer = tokenizer
            self.is_prepared = True
            return

        # Add new tokens to tokenizer
        tokenizer.add_tokens(sorted(self.get_vocab()))  # NOTE: set is unordered in python
        logger.warning(f"Adding {len(self.get_vocab())} new tokens to tokenizer")
        if model is not None:
            model.resize_token_embeddings(len(tokenizer))

            if apply_semantic_init:
                apply_semantic_initialization(tokenizer, model, self.config.encoder_type)

        self.tokenizer = tokenizer
        self.is_prepared = True

    @overload
    def tokenize_event(
        self,
        mcap_msg: McapMessage,
        *,
        return_dict: Literal[True] = True,
    ) -> TokenizedEvent: ...

    @overload
    def tokenize_event(
        self,
        mcap_msg: McapMessage,
        *,
        return_dict: Literal[False],
    ) -> npt.NDArray[np.int64]: ...

    def tokenize_event(
        self,
        mcap_msg: McapMessage,
        *,
        return_dict: bool = True,
    ) -> TokenizedEvent | npt.NDArray[np.int64]:
        if not self.is_prepared:
            raise RuntimeError("EpisodeTokenizer must be prepared by `prepare_model` before tokenizing")

        encoded_text, images = self.encoder.encode(mcap_msg)

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
    ) -> Iterator[TokenizedEvent]:
        for mcap_msg in mcap_messages:
            yield self.tokenize_event(mcap_msg)

    def decode_episode(
        self,
        input_ids_or_text: list[int] | npt.NDArray[np.int64] | str,
        *,
        skip_invalid: bool = True,
        adjust_timestamp: bool = True,
    ) -> Iterator[McapMessage]:
        """Decode token IDs or tokenized text back to the original McapMessage format."""
        if not isinstance(self.encoder, (HierarchicalEventEncoder, FactorizedEventEncoder)):
            raise NotImplementedError(
                f"EpisodeTokenizer.decode_episode is only implemented for HierarchicalEventEncoder and FactorizedEventEncoder, "
                f"got {type(self.encoder)}"
            )

        # Convert token IDs back to text (if input is token IDs)
        if isinstance(input_ids_or_text, str):
            text = input_ids_or_text
        else:
            if not self.is_prepared:
                raise RuntimeError("EpisodeTokenizer must be prepared by `prepare_model` before decoding")
            # Input is token IDs
            text = self.tokenizer.decode(input_ids_or_text, skip_special_tokens=False)

        # Parse all events between <EVENT_START> and <EVENT_END> tokens
        event_strings = re.findall(r"<EVENT_START>.*?<EVENT_END>", text)

        # Initialize previous timestamp and timestamp bias
        previous_timestamp = float("-inf")
        timestamp_bias = 0
        for event_string in event_strings:
            try:
                # Convert repeated image token sequences back to fake_image_placeholder
                # Pattern: prefix + (image_token * image_token_length) + suffix -> fake_image_placeholder
                processed_event_string = event_string.replace(
                    f"{self.config.image_token_prefix}{self.config.image_token_suffix}",
                    self.config.fake_image_placeholder,
                )
                event = self.encoder.decode(processed_event_string)
                # Handle timestamp adjustment for both encoder types
                if adjust_timestamp:
                    # Get timestamp range from encoder config (both encoders have this property)
                    timestamp_range = self.encoder.config.timestamp_range

                    # Adjust timestamp if it's smaller than the previous one (modular arithmetic)
                    if event.timestamp < previous_timestamp:
                        timestamp_bias += timestamp_range

                    # Apply timestamp bias
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
        def process_event(event):
            mcap_message = McapMessage.model_validate_json(event["mcap_message"])
            tokenized_event = self.tokenize_event(mcap_message)

            return {
                "episode_path": event["episode_path"],
                "topic": event["topic"],
                "timestamp_ns": event["timestamp_ns"],
                "text": tokenized_event["text"],
                "images": [image.model_dump_json() for image in tokenized_event["images"]],
                "token_ids": tokenized_event["token_ids"],
                "total_token_count": tokenized_event["total_token_count"],
            }

        # Tokenize the dataset
        tokenized_dataset = event_dataset.map(
            process_event,
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
