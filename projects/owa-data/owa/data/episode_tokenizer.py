from dataclasses import dataclass
from typing import Iterator, TypedDict

from transformers import AutoTokenizer, PreTrainedTokenizer

from mcap_owa.highlevel import McapMessage
from owa.data.encoders import create_encoder
from owa.msgs.desktop.screen import ScreenCaptured


@dataclass
class EpisodeTokenizerConfig:
    encoder_type: str = "hierarchical"
    image_token_length: int = 64
    image_token: str = "<IMAGE>"
    episode_start_token: str = "<EPISODE_START>"
    episode_end_token: str = "<EPISODE_END>"


class TokenizedEpisode(TypedDict):
    token_ids: list[int]
    images: list[ScreenCaptured]
    total_token_count: int


class EpisodeTokenizer:
    def __init__(self, config: EpisodeTokenizerConfig = EpisodeTokenizerConfig(), **kwargs):
        self.config = EpisodeTokenizerConfig(**(config.__dict__ | kwargs))
        self.encoder = create_encoder(self.config.encoder_type, image_token=self.config.image_token)
        self.is_prepared = False

    def get_vocab(self) -> set[str]:
        return self.encoder.get_vocab() | {
            self.config.image_token,
            self.config.episode_start_token,
            self.config.episode_end_token,
        }

    def prepare_model(self, *, tokenizer: PreTrainedTokenizer, model=None):
        tokenizer.add_tokens(list(self.get_vocab()))
        if model is not None:
            model.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer
        self.is_prepared = True

    def tokenize(self, episode_iterator: Iterator[McapMessage]) -> Iterator[TokenizedEpisode]:
        if not self.is_prepared:
            raise RuntimeError("EpisodeTokenizer must be prepared by `prepare_model` before tokenizing")

        for mcap_msg in episode_iterator:
            encoded_text, images = self.encoder.encode(mcap_msg)
            token_ids = self.tokenizer.encode(encoded_text, add_special_tokens=False)

            yield TokenizedEpisode(
                token_ids=token_ids,
                images=images,
                total_token_count=len(token_ids) + (self.config.image_token_length - 1) * len(images),
            )
