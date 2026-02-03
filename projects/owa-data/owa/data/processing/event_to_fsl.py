from dataclasses import dataclass, field
from typing import Union

from loguru import logger
from transformers import AutoTokenizer

from owa.data.collator import ModelType, detect_model_type
from owa.data.datasets import Dataset, DatasetDict, DatasetStage
from owa.data.encoders import create_encoder
from owa.data.tokenization import (
    EventTokenizationContext,
    ImageTokenConfig,
    expand_tokenizer_for_events,
    tokenize_event_dataset,
)

from .fsl_processing import FSLDatasetConfig, precompute_fsl_dataset


@dataclass
class EventToFSLConfig:
    """Configuration for event to FSL conversion."""

    # Model configuration
    tokenizer_name: str

    # Tokenization configuration
    encoder_type: str = "factorized"
    image_token_config: ImageTokenConfig | None = None  # None means auto-detect from tokenizer_name

    # FSL configuration
    fsl_dataset: FSLDatasetConfig = field(default_factory=FSLDatasetConfig)

    # Processing options
    num_proc: int = 32  # Number of processes for tokenization
    fsl_workers: int = 4  # Number of workers for FSL processing


def _get_image_config_for_model(model_name_or_path: str, config: EventToFSLConfig) -> ImageTokenConfig:
    """Get ImageTokenConfig based on model type or explicit config."""
    # If explicitly provided, use it
    if config.image_token_config is not None:
        return config.image_token_config

    # Otherwise, detect from model type
    model_type = detect_model_type(model_name_or_path)

    if model_type == ModelType.INTERNVL:
        return ImageTokenConfig(
            prefix="<img>",
            token="<IMG_CONTEXT>",
            length=256,
            suffix="</img>",
        )
    else:
        # SmolVLM2 and other models
        return ImageTokenConfig(
            prefix="<fake_token_around_image><global-img>",
            token="<image>",
            length=64,
            suffix="<fake_token_around_image>",
        )


def build_fsl_dataset(
    event_dataset: Union[Dataset, DatasetDict], *, config: EventToFSLConfig
) -> Union[Dataset, DatasetDict]:
    """
    Convert event dataset to FSL (Fixed Sequence Length) dataset format.

    Args:
        event_dataset: Input event dataset (Dataset or DatasetDict)
        config: Configuration for the conversion process

    Returns:
        FSL dataset (Dataset or DatasetDict depending on input)

    Raises:
        ValueError: If input dataset is not EVENT stage
        RuntimeError: If tokenizer preparation fails
    """
    logger.info("Starting event to FSL dataset conversion")
    logger.info(f"Tokenizer: {config.tokenizer_name}")
    logger.info(f"Encoder type: {config.encoder_type}")
    logger.info(f"FSL dataset config: {config.fsl_dataset}")

    ds_dict = event_dataset

    if isinstance(ds_dict, DatasetDict):
        logger.info(f"Loaded DatasetDict with splits: {list(ds_dict.keys())}")
        splits = list(ds_dict.keys())
    else:
        logger.info("Loaded single Dataset")
        splits = [None]

    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create encoder and image config
    image_config = _get_image_config_for_model(config.tokenizer_name, config)
    encoder = create_encoder(config.encoder_type, fake_image_placeholder=image_config.fake_placeholder)

    # Expand tokenizer with event tokens
    expand_tokenizer_for_events(tokenizer, encoder, image_config)

    # Create tokenization context
    ctx = EventTokenizationContext(encoder=encoder, tokenizer=tokenizer, image_config=image_config)
    logger.info(f"Created tokenization context with {ctx.encoder_type} encoder")

    # Configure FSL dataset
    config.fsl_dataset.pad_token_id = tokenizer.pad_token_id
    logger.info(f"FSL dataset config: {config.fsl_dataset}")

    processed_datasets = {}

    for split in splits:
        ds = ds_dict[split] if split else ds_dict

        if ds.owa_config.stage != DatasetStage.EVENT:
            raise ValueError(f"Input dataset must be EVENT stage, got {ds.owa_config.stage}")

        split_name = split if split else "train"
        logger.info(f"Processing {len(ds):,} events from {split_name} split")

        # Step 1: Tokenize event dataset
        logger.info(f"Tokenizing {split_name} events...")
        tokenized_dataset = tokenize_event_dataset(ctx, ds, map_kwargs={"num_proc": config.num_proc})
        logger.info(f"Created {len(tokenized_dataset):,} tokenized events")

        # Step 2: Create FSL dataset
        logger.info("Creating FSL dataset from tokenized events...")
        fsl_dataset = precompute_fsl_dataset(
            tokenized_dataset, config=config.fsl_dataset, num_workers=config.fsl_workers
        )
        fsl_dataset.owa_config.event_to_fsl_config = config.__dict__
        logger.info(f"Created {len(fsl_dataset):,} FSL sequences for {split_name} split")

        processed_datasets[split_name] = fsl_dataset

    # Combine into DatasetDict if multiple splits
    final_dataset = (
        DatasetDict(processed_datasets) if len(processed_datasets) > 1 else list(processed_datasets.values())[0]
    )

    # Log summary
    if len(processed_datasets) > 1:
        total_sequences = sum(len(ds) for ds in processed_datasets.values())
        logger.info(f"Created {total_sequences:,} total FSL sequences")
        for split_name, ds in processed_datasets.items():
            logger.info(f"  {split_name}: {len(ds):,} sequences")
    else:
        split_name = list(processed_datasets.keys())[0]
        ds = list(processed_datasets.values())[0]
        logger.info(f"Created {len(ds):,} FSL sequences ({split_name})")

    logger.info("FSL dataset creation completed successfully!")
    return final_dataset
