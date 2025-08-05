from dataclasses import dataclass

from datasets import Dataset as HFDataset
from loguru import logger

from .config import DatasetConfig, DatasetStage
from .dataset import Dataset


@dataclass
class FSLDatasetConfig:
    pad_token_id: int = 0
    max_sequence_length: int = 8192


def precompute_fsl_dataset(
    tokenized_dataset: Dataset,
    config: FSLDatasetConfig = FSLDatasetConfig(),
    **kwargs,
) -> Dataset:
    """
    Pre-compute FSL dataset from tokenized dataset, excluding image loading.

    Args:
        tokenized_dataset: Input tokenized dataset
        config: FSL dataset configuration
        **kwargs: Additional config parameters

    Returns:
        Pre-computed FSL dataset
    """
    config = FSLDatasetConfig(**(config.__dict__ | kwargs))

    logger.info(f"Pre-computing FSL sequences from {len(tokenized_dataset)} events")

    def pad_and_yield(tokens, texts, images, episode_path):
        """Helper function to pad tokens and yield a sequence."""
        padded_tokens = tokens + [config.pad_token_id] * (config.max_sequence_length - len(tokens))
        attention_mask = [1] * len(tokens) + [0] * (config.max_sequence_length - len(tokens))

        return {
            "input_ids": padded_tokens,
            "attention_mask": attention_mask,
            "texts": "".join(texts),
            "images": images,
            "episode_path": episode_path,
        }

    def sequence_generator():
        """Generator that yields fixed-length sequences by accumulating tokens from events."""
        current_tokens = []
        current_texts = []
        current_images = []
        current_episode_path = None

        for event in tokenized_dataset:
            event_tokens = list(event["token_ids"])
            event_text = event["text"]
            event_images = list(event["images"])
            event_episode_path = event["episode_path"]

            if len(event_tokens) > config.max_sequence_length:
                logger.warning(
                    f"Skipping an event of length {len(event_tokens)} because it is longer than max_sequence_length={config.max_sequence_length}"
                )
                continue

            # Check if adding this event would exceed max_sequence_length or change the episode
            if len(current_tokens) + len(event_tokens) > config.max_sequence_length or (
                current_episode_path is not None and current_episode_path != event_episode_path
            ):
                # Yield current sequence
                yield pad_and_yield(current_tokens, current_texts, current_images, current_episode_path)

                # Start new sequence
                current_tokens = []
                current_texts = []
                current_images = []
                current_episode_path = None

            current_tokens.extend(event_tokens)
            current_texts.append(event_text)
            current_images.extend(event_images)
            current_episode_path = event_episode_path

        # Yield final sequence if it has tokens
        if current_tokens:
            yield pad_and_yield(current_tokens, current_texts, current_images, current_episode_path)

    # Create HuggingFace dataset from generator
    hf_dataset = HFDataset.from_generator(sequence_generator)

    # Create OWA Dataset with FSL stage
    owa_config = DatasetConfig(
        stage=DatasetStage.FSL, mcap_root_directory=tokenized_dataset.owa_config.mcap_root_directory
    )

    fsl_dataset = Dataset.from_hf_dataset(hf_dataset, owa_config=owa_config)

    logger.info(f"Pre-computed FSL dataset with {len(fsl_dataset)} sequences")
    return fsl_dataset
