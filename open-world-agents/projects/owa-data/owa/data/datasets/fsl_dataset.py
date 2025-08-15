from dataclasses import dataclass

from loguru import logger

from .config import DatasetConfig, DatasetStage
from .dataset import Dataset


@dataclass
class FSLDatasetConfig:
    """Configuration for FSL dataset processing."""

    pad_token_id: int = 0
    max_sequence_length: int = 8192


def _process_batch_to_sequences(batch, config: FSLDatasetConfig):
    """Process a batch of events into FSL sequences using datasets.map."""

    def pad_sequence(tokens, texts, images, episode_path):
        padded_tokens = tokens + [config.pad_token_id] * (config.max_sequence_length - len(tokens))
        attention_mask = [1] * len(tokens) + [0] * (config.max_sequence_length - len(tokens))
        return {
            "input_ids": padded_tokens,
            "attention_mask": attention_mask,
            "texts": "".join(texts),
            "images": images,
            "episode_path": episode_path,
        }

    sequences = []
    current_tokens, current_texts, current_images, current_episode_path = [], [], [], None

    # Process all events in the batch
    for i in range(len(batch["token_ids"])):
        event_tokens = list(batch["token_ids"][i])
        event_text = batch["text"][i]
        event_images = list(batch["images"][i])
        event_episode_path = batch["episode_path"][i]

        if len(event_tokens) > config.max_sequence_length:
            logger.warning(
                f"Skipping an event of {len(event_tokens)=} because it is longer than {config.max_sequence_length=}"
            )
            continue

        if len(current_tokens) + len(event_tokens) > config.max_sequence_length or (
            current_episode_path is not None and current_episode_path != event_episode_path
        ):
            if current_tokens:
                sequences.append(pad_sequence(current_tokens, current_texts, current_images, current_episode_path))
            current_tokens, current_texts, current_images, current_episode_path = [], [], [], None

        current_tokens.extend(event_tokens)
        current_texts.append(event_text)
        current_images.extend(event_images)
        current_episode_path = event_episode_path

    if current_tokens:
        sequences.append(pad_sequence(current_tokens, current_texts, current_images, current_episode_path))

    # Return in the format expected by datasets.map (flattened)
    if not sequences:
        return {
            "input_ids": [],
            "attention_mask": [],
            "texts": [],
            "images": [],
            "episode_path": [],
        }

    # Flatten sequences into separate lists for each field
    return {
        "input_ids": [seq["input_ids"] for seq in sequences],
        "attention_mask": [seq["attention_mask"] for seq in sequences],
        "texts": [seq["texts"] for seq in sequences],
        "images": [seq["images"] for seq in sequences],
        "episode_path": [seq["episode_path"] for seq in sequences],
    }


def precompute_fsl_dataset(
    tokenized_dataset: Dataset,
    config: FSLDatasetConfig = FSLDatasetConfig(),
    num_workers: int = 4,
    batch_size: int = 65536,
    **kwargs,
) -> Dataset:
    """
    Pre-compute FSL dataset using HuggingFace datasets.map with batching.

    Args:
        tokenized_dataset: Input tokenized dataset
        config: FSL dataset configuration
        num_workers: Number of parallel workers (0 = sequential)
        batch_size: Batch size for processing
        **kwargs: Additional config parameters

    Returns:
        Pre-computed FSL dataset
    """
    config = FSLDatasetConfig(**(config.__dict__ | kwargs))
    logger.info(
        f"Pre-computing FSL sequences using datasets.map with batch_size={batch_size:,}, num_workers={num_workers}"
    )

    def process_batch(batch):
        return _process_batch_to_sequences(batch, config)

    # Use datasets.map with batching
    mapped_dataset = tokenized_dataset.map(
        process_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_workers if num_workers > 0 else None,
        remove_columns=tokenized_dataset.column_names,
    )

    # Create OWA Dataset with FSL stage
    owa_config = DatasetConfig(
        stage=DatasetStage.FSL, mcap_root_directory=tokenized_dataset.owa_config.mcap_root_directory
    )

    fsl_dataset = Dataset.from_hf_dataset(mapped_dataset, owa_config=owa_config)

    logger.info(f"Pre-computed FSL dataset with {len(fsl_dataset)} sequences")
    return fsl_dataset
