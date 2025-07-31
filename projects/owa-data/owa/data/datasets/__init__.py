"""OWA Datasets - Unified HuggingFace Dataset implementation with stage-specific functionality."""

from .config import DatasetConfig, DatasetStage
from .dataset import Dataset, DatasetDict
from .discovery import list_datasets
from .fsl_dataset import (
    FSLDatasetConfig,
    precompute_fsl_dataset,
)
from .load import load_dataset, load_from_disk
from .transforms import (
    create_binned_transform,
    create_event_transform,
    create_fsl_transform,
    create_tokenized_transform,
    create_transform,
)

__all__ = [
    # Core Dataset Classes
    "Dataset",
    "DatasetDict",
    "load_dataset",
    "load_from_disk",
    # Configuration
    "DatasetConfig",
    "DatasetStage",
    "FSLDatasetConfig",
    # Main Functions
    "list_datasets",
    "precompute_fsl_dataset",
    # Transform Functions
    "create_event_transform",
    "create_binned_transform",
    "create_fsl_transform",
    "create_tokenized_transform",
    "create_transform",
]
