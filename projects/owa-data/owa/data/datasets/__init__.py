"""OWA Datasets - Custom HuggingFace Dataset implementations with OWA-specific functionality."""

# Core dataset classes
from .base import OWADatasetBase
from .binned import BinnedDataset

# Configuration classes
from .config import (
    BinnedDatasetConfig,
    DatasetType,
    EventDatasetConfig,
    FSLDatasetConfig,
    OWADatasetConfig,
)

# Utility functions
from .discovery import list_datasets
from .event import EventDataset
from .fsl import FSLDataset
from .utils import create_dataset, load_dataset

__all__ = [
    # Dataset Types
    "DatasetType",
    # Configuration Classes
    "OWADatasetConfig",
    "EventDatasetConfig",
    "BinnedDatasetConfig",
    "FSLDatasetConfig",
    # Dataset Classes
    "OWADatasetBase",
    "EventDataset",
    "BinnedDataset",
    "FSLDataset",
    # Main Functions
    "load_dataset",
    "create_dataset",
    "list_datasets",
]
