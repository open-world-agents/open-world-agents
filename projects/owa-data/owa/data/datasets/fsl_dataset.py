from dataclasses import dataclass, field

from loguru import logger

from .config import DatasetConfig, DatasetStage
from .dataset import Dataset


@dataclass
class FSLDatasetConfig:
    """Configuration for FSL dataset processing."""

    pad_token_id: int = 0
    max_sequence_length: int = 8192
    include_samples_without_images: bool = False  # Whether to include samples that don't contain images when tokenized

    # Topics to apply time shift to
    action_topics: list[str] = field(default_factory=lambda: ["keyboard", "mouse/raw"])
    skip_first_t_seconds_for_action: float | None = None  # Skip the first t seconds of action topics per sample
    time_shift_seconds_for_action: float | None = None  # Time shift in seconds to add to action topics


def _process_batch_to_sequences(batch, config: FSLDatasetConfig):
    """Process a batch of tokenized events into FSL sequences."""

    def pad_sequence(items):
        # Skip the first t seconds of action topics if specified. NOTE: order of skip and time shift is important
        if config.skip_first_t_seconds_for_action is not None:
            first_action_timestamp = min(
                items["timestamp_ns"][i] for i, topic in enumerate(items["topic"]) if topic not in config.action_topics
            )
            skip_threshold = first_action_timestamp + int(config.skip_first_t_seconds_for_action * 1e9)
            before_skip = len(items["timestamp_ns"])
            items = {
                key: [
                    val
                    for i, val in enumerate(items[key])
                    if (items["topic"][i] not in config.action_topics or items["timestamp_ns"][i] >= skip_threshold)
                ]
                for key in items.keys()
            }
            logger.debug(f"Skipped {before_skip - len(items['timestamp_ns'])} action events")

        # Apply time shift to action topics if specified. NOTE: order of skip and time shift is important
        if config.time_shift_seconds_for_action is not None:
            for i, topic in enumerate(items["topic"]):
                if topic in config.action_topics:
                    items["timestamp_ns"][i] += int(config.time_shift_seconds_for_action * 1e9)

            # Sort by timestamp
            sorted_items = sorted((val, i) for i, val in enumerate(items["timestamp_ns"]))
            items = {key: [val[i] for _, i in sorted_items] for key, val in items.items()}
            logger.debug(f"Time shifted {len(items['timestamp_ns'])} action events")

        tokens = sum(items["token_ids"], [])
        texts = "".join(items["text"])
        images = sum(items["images"], [])
        episode_path = items["episode_path"][0]

        # Calculate attention mask and pad tokens to max sequence length
        padded_tokens = tokens + [config.pad_token_id] * (config.max_sequence_length - len(tokens))
        attention_mask = [1] * len(tokens) + [0] * (config.max_sequence_length - len(tokens))

        return {
            "input_ids": padded_tokens,
            "attention_mask": attention_mask,
            "texts": texts,
            "images": images,
            "episode_path": episode_path,
        }

    sequences = []
    items, current_tokens_count, current_episode_path = {}, 0, None

    # Track filtering statistics
    skipped_no_images = 0
    skipped_too_long = 0

    # Process all events in the batch
    for i in range(len(batch["token_ids"])):
        item = {
            "episode_path": batch["episode_path"][i],
            "topic": batch["topic"][i],
            "timestamp_ns": batch["timestamp_ns"][i],
            "text": batch["text"][i],
            "images": batch["images"][i],
            "token_ids": batch["token_ids"][i],
            "total_token_count": batch["total_token_count"][i],
        }

        if len(item["token_ids"]) > config.max_sequence_length:
            skipped_too_long += 1
            continue

        if current_tokens_count + len(item["token_ids"]) > config.max_sequence_length or (
            current_episode_path is not None and current_episode_path != item["episode_path"]
        ):
            if current_tokens_count:
                if not config.include_samples_without_images and len(items["images"]) == 0:
                    skipped_no_images += 1
                else:
                    sequences.append(pad_sequence(items))
            items = {}
            current_tokens_count = 0
            current_episode_path = None

        for key in item.keys():
            items.setdefault(key, []).append(item[key])
        current_episode_path = item["episode_path"]
        current_tokens_count += len(item["token_ids"])

    if current_tokens_count:
        if not config.include_samples_without_images and len(items["images"]) == 0:
            skipped_no_images += 1
        else:
            sequences.append(pad_sequence(items))

    # Log filtering statistics if any events were skipped
    if skipped_too_long > 0 or skipped_no_images > 0:
        logger.info(
            f"Batch processing: {len(sequences)} sequences processed "
            f"(skipped {skipped_too_long} too long, {skipped_no_images} without images)"
        )

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
