"""OWA Datasets - Unified HuggingFace Dataset implementation with stage-specific functionality."""

from .config import DatasetConfig, DatasetStage
from .dataset import Dataset, DatasetDict
from .discovery import list_datasets
from .fsl_dataset import FSLDataset, FSLDatasetConfig, prepare_fsl
from .load import load_from_disk
from .transforms import (
    create_binned_transform,
    create_event_transform,
    create_tokenized_transform,
    create_transform,
)

__all__ = [
    # Core Dataset Classes
    "Dataset",
    "DatasetDict",
    "FSLDataset",
    "load_from_disk",
    # Configuration
    "DatasetConfig",
    "DatasetStage",
    "FSLDatasetConfig",
    # Main Functions
    "list_datasets",
    "prepare_fsl",
    # Transform Functions
    "create_event_transform",
    "create_binned_transform",
    "create_tokenized_transform",
    "create_transform",
]
