from dataclasses import dataclass
from typing import TypedDict

from transformers import PreTrainedTokenizer

from mcap_owa.highlevel import McapMessage
from owa.data.encoders import create_encoder
from owa.msgs.desktop.screen import ScreenCaptured

from .datasets import Dataset, DatasetStage


@dataclass
class EpisodeTokenizerConfig:
    encoder_type: str = "hierarchical"
    image_token_length: int = 64
    image_token: str = "<image>"
    image_token_prefix: str = "<fake_token_around_image><global-img>"
    image_token_suffix: str = "<fake_token_around_image>"
    episode_start_token: str = "<EPISODE_START>"
    episode_end_token: str = "<EPISODE_END>"


class TokenizedEvent(TypedDict):
    text: str
    token_ids: list[int]
    images: list[ScreenCaptured]
    total_token_count: int


class EpisodeTokenizer:
    def __init__(self, config: EpisodeTokenizerConfig = EpisodeTokenizerConfig(), **kwargs):
        self.config = EpisodeTokenizerConfig(**(config.__dict__ | kwargs))
        self.encoder = create_encoder(
            self.config.encoder_type,
            image_token=self.config.image_token,
            image_token_prefix=self.config.image_token_prefix,
            image_token_suffix=self.config.image_token_suffix,
        )
        self.is_prepared = False

    def get_vocab(self) -> set[str]:
        return self.encoder.get_vocab() | {
            self.config.image_token,
            self.config.image_token_prefix,
            self.config.image_token_suffix,
            self.config.episode_start_token,
            self.config.episode_end_token,
        }

    def prepare_model(self, *, tokenizer: PreTrainedTokenizer, model=None):
        tokenizer.add_tokens(sorted(self.get_vocab()))  # note: set is unordered in python
        if model is not None:
            model.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer
        self.is_prepared = True

    def tokenize(self, mcap_msg: McapMessage) -> TokenizedEvent:
        if not self.is_prepared:
            raise RuntimeError("EpisodeTokenizer must be prepared by `prepare_model` before tokenizing")

        encoded_text, images = self.encoder.encode(mcap_msg)
        # Replace the image token sequence with repeated image tokens
        # First, create the pattern to search for
        image_token_pattern = (
            f"{self.config.image_token_prefix}{self.config.image_token}{self.config.image_token_suffix}"
        )
        # Replace with prefix + repeated tokens + suffix
        replacement = f"{self.config.image_token_prefix}{self.config.image_token * self.config.image_token_length}{self.config.image_token_suffix}"
        encoded_text = encoded_text.replace(image_token_pattern, replacement)
        token_ids = self.tokenizer.encode(encoded_text, add_special_tokens=False)

        return TokenizedEvent(
            text=encoded_text,
            token_ids=token_ids,
            images=images,
            total_token_count=len(token_ids),
        )

    def tokenize_event_dataset(self, event_dataset: Dataset, map_kwargs: dict = {"num_proc": 16}) -> Dataset:
        def process_event(event, idx):
            prefix_text = suffix_text = ""
            # Add episode start token
            if idx == 0 or (idx > 0 and event["episode_path"] != event_dataset[idx - 1]["episode_path"]):
                prefix_text = self.config.episode_start_token
            # Add episode end token
            if idx < len(event_dataset) - 1 and event["episode_path"] != event_dataset[idx + 1]["episode_path"]:
                suffix_text = self.config.episode_end_token

            prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
            suffix_ids = self.tokenizer.encode(suffix_text, add_special_tokens=False)

            episode_path = event["episode_path"]
            mcap_message = McapMessage.model_validate_json(event["mcap_message"])
            tokenized_event = self.tokenize(mcap_message)

            tokenized_event["text"] = f"{prefix_text}{tokenized_event['text']}{suffix_text}"
            tokenized_event["token_ids"] = prefix_ids + tokenized_event["token_ids"] + suffix_ids
            tokenized_event["total_token_count"] += len(prefix_ids) + len(suffix_ids)

            return {
                "episode_path": episode_path,
                "text": tokenized_event["text"],
                "token_ids": tokenized_event["token_ids"],
                "images": [image.model_dump_json() for image in tokenized_event["images"]],
                "total_token_count": tokenized_event["total_token_count"],
            }

        tokenized_dataset = event_dataset.map(
            process_event,
            with_indices=True,
            desc="Tokenizing event dataset",
            remove_columns=event_dataset.column_names,
            **map_kwargs,
        )
        tokenized_dataset = Dataset.from_hf_dataset(tokenized_dataset, owa_config=event_dataset.owa_config)
        tokenized_dataset.owa_config.stage = DatasetStage.TOKENIZED

        return tokenized_dataset
