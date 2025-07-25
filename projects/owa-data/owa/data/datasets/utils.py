"""Utility functions for dataset loading and creation."""

from typing import Optional

from datasets import Dataset as HFDataset
from datasets.utils.typing import PathLike

from .base import OWADatasetBase
from .binned import BinnedDataset
from .config import DatasetType, OWADatasetConfig
from .discovery import resolve_dataset_path_and_config
from .event import EventDataset
from .fsl import FSLDataset


def load_dataset(dataset_path: PathLike, storage_options: Optional[dict] = None, **kwargs) -> OWADatasetBase:
    """
    Load an OWA dataset from disk with automatic type detection.

    This function mimics the behavior of HuggingFace's load_dataset but for OWA datasets.
    It automatically detects the dataset type from the saved config and returns the
    appropriate dataset class. Supports both local and remote filesystems.

    Args:
        dataset_path: Path to the saved dataset directory (local or remote URI)
        storage_options: Key/value pairs for remote filesystem backend
        **kwargs: Additional arguments passed to the dataset constructor

    Returns:
        The appropriate OWA dataset instance (EventDataset, BinnedDataset, etc.)

    Example:
        ```python
        from owa.data.datasets import load_dataset

        # Local dataset
        dataset = load_dataset("/path/to/saved/dataset")

        # Remote dataset (S3, GCS, etc.)
        dataset = load_dataset("s3://bucket/dataset", storage_options={"key": "value"})

        # HuggingFace Hub dataset
        dataset = load_dataset("hf://datasets/username/dataset-name")

        print(f"Loaded: {type(dataset).__name__}")  # EventDataset, BinnedDataset, etc.
        ```
    """
    # Resolve path and load config with remote filesystem support
    _, config_data, _ = resolve_dataset_path_and_config(dataset_path, storage_options)

    # Extract storage_options from kwargs to pass to HF load_from_disk
    hf_kwargs = kwargs.copy()
    if storage_options:
        hf_kwargs["storage_options"] = storage_options

    if config_data:
        try:
            dataset_type = DatasetType(config_data.get("dataset_type"))

            if dataset_type == DatasetType.EVENT:
                return EventDataset.load_from_disk(dataset_path, **hf_kwargs)
            elif dataset_type == DatasetType.BINNED:
                return BinnedDataset.load_from_disk(dataset_path, **hf_kwargs)
            elif dataset_type == DatasetType.FSL:
                return FSLDataset.load_from_disk(dataset_path, **hf_kwargs)
        except Exception as e:
            print(f"Warning: Could not load OWA dataset type from config: {e}")

    # Fallback to base OWA dataset if no config or unknown type
    hf_dataset = HFDataset.load_from_disk(dataset_path, **hf_kwargs)
    return OWADatasetBase(
        arrow_table=hf_dataset.data,
        info=hf_dataset.info,
        split=hf_dataset.split,
        indices_table=getattr(hf_dataset, "_indices", None),
        fingerprint=getattr(hf_dataset, "_fingerprint", None),
    )


def create_dataset(
    dataset_type: DatasetType, *args, owa_config: Optional[OWADatasetConfig] = None, **kwargs
) -> OWADatasetBase:
    """
    Create an OWA dataset of the specified type.

    Args:
        dataset_type: Type of dataset to create
        *args: Arguments passed to the dataset constructor
        owa_config: Configuration for the dataset
        **kwargs: Additional keyword arguments

    Returns:
        The created dataset instance

    Example:
        ```python
        from owa.data.datasets import create_dataset, DatasetType, EventDatasetConfig

        config = EventDatasetConfig("/data", DatasetType.EVENT)
        dataset = create_dataset(DatasetType.EVENT, data, owa_config=config)
        ```
    """
    if dataset_type == DatasetType.EVENT:
        from .config import EventDatasetConfig

        config = owa_config if isinstance(owa_config, EventDatasetConfig) else None
        return EventDataset(*args, owa_config=config, **kwargs)
    elif dataset_type == DatasetType.BINNED:
        from .config import BinnedDatasetConfig

        config = owa_config if isinstance(owa_config, BinnedDatasetConfig) else None
        return BinnedDataset(*args, owa_config=config, **kwargs)
    elif dataset_type == DatasetType.FSL:
        from .config import FSLDatasetConfig

        config = owa_config if isinstance(owa_config, FSLDatasetConfig) else None
        return FSLDataset(*args, owa_config=config, **kwargs)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
