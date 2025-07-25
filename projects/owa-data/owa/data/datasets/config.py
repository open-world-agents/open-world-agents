"""Dataset configuration classes."""

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, Optional

from datasets.utils.typing import PathLike


class DatasetType(StrEnum):
    EVENT = "event"
    BINNED = "binned"
    FSL = "fsl"


@dataclass
class OWADatasetConfig:
    """Base configuration for OWA datasets."""

    mcap_root_directory: PathLike
    dataset_type: DatasetType

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OWADatasetConfig":
        return cls(**data)

    @classmethod
    def from_json(cls, path: PathLike) -> "OWADatasetConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_json(self, path: PathLike) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)


@dataclass
class EventDatasetConfig(OWADatasetConfig):
    """Configuration for event datasets."""

    dataset_type: DatasetType = DatasetType.EVENT
    rate_settings: Optional[Dict[str, float]] = None
    keep_topics: Optional[list[str]] = None
    num_workers: int = 4
    source_train_dir: Optional[str] = None
    source_test_dir: Optional[str] = None
    test_percent: Optional[float] = None
    encoder_type: str = "hierarchical"
    load_images: bool = True


@dataclass
class BinnedDatasetConfig(OWADatasetConfig):
    """Configuration for binned datasets."""

    dataset_type: DatasetType = DatasetType.BINNED
    fps: float = 10.0
    filter_empty_actions: bool = False
    bin_interval_ns: Optional[int] = None
    source_event_dataset: Optional[str] = None
    instruction: str = "Complete the computer task"
    encoder_type: str = "hierarchical"
    load_images: bool = True


@dataclass
class FSLDatasetConfig(OWADatasetConfig):
    """Configuration for FSL (Fixed Sequence Length) datasets."""

    dataset_type: DatasetType = DatasetType.FSL
    pad_token_id: int = 0
    max_sequence_length: int = 8192
    load_images: bool = True
