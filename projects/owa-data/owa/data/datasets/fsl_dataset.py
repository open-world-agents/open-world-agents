import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from datasets import Dataset as HFDataset
from datasets import concatenate_datasets
from loguru import logger

from .config import DatasetConfig, DatasetStage
from .dataset import Dataset


@dataclass
class FSLDatasetConfig:
    """Configuration for FSL dataset processing."""

    pad_token_id: int = 0
    max_sequence_length: int = 8192


def _process_events_to_sequences(dataset: HFDataset, config: FSLDatasetConfig) -> HFDataset:
    """Process events into FSL sequences."""

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

        for event in dataset:
            event_tokens = list(event["token_ids"])
            event_text = event["text"]
            event_images = list(event["images"])
            event_episode_path = event["episode_path"]

            if len(event_tokens) > config.max_sequence_length:
                logger.warning(
                    f"Skipping an event of {len(event_tokens)=} because it is longer than {config.max_sequence_length=}"
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

    return HFDataset.from_generator(sequence_generator)


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


def precompute_fsl_dataset_legacy(
    tokenized_dataset: Dataset,
    config: FSLDatasetConfig = FSLDatasetConfig(),
    num_workers: int = 4,
    chunk_size: int = 65536,
    **kwargs,
) -> Dataset:
    """
    Pre-compute FSL dataset from tokenized dataset using parallel processing.

    Args:
        tokenized_dataset: Input tokenized dataset
        config: FSL dataset configuration
        num_workers: Number of parallel workers (1 = sequential, >1 = parallel)
        chunk_size: Size of each chunk to process in parallel
        **kwargs: Additional config parameters

    Returns:
        Pre-computed FSL dataset
    """
    config = FSLDatasetConfig(**(config.__dict__ | kwargs))
    logger.info(f"Pre-computing FSL sequences from {len(tokenized_dataset)} events using {num_workers} workers")

    if num_workers == 0:
        hf_dataset = _process_events_to_sequences(tokenized_dataset, config)
    else:
        num_shards = min(num_workers, math.ceil(len(tokenized_dataset) / chunk_size))
        hf_datasets = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for shard_idx in range(num_shards):
                shard = tokenized_dataset.shard(num_shards=num_shards, index=shard_idx)
                futures.append(executor.submit(_process_events_to_sequences, shard, config))

            for future in as_completed(futures):
                hf_datasets.append(future.result())

        hf_dataset = concatenate_datasets(hf_datasets)

    # Create OWA Dataset with FSL stage
    owa_config = DatasetConfig(
        stage=DatasetStage.FSL, mcap_root_directory=tokenized_dataset.owa_config.mcap_root_directory
    )
    fsl_dataset = Dataset.from_hf_dataset(hf_dataset, owa_config=owa_config)

    logger.info(f"Pre-computed FSL dataset with {len(fsl_dataset)} sequences")
    return fsl_dataset


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
