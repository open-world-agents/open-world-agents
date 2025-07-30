"""Dataset configuration classes."""

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Dict, Optional

from datasets.utils.typing import PathLike


class DatasetStage(StrEnum):
    """Dataset processing stages in the OWA pipeline."""

    EVENT = "event"  # Raw MCAP events
    BINNED = "binned"  # Time-binned events (state/action sequences)
    TOKENIZED = "tokenized"  # Tokenized events from EpisodeTokenizer
    FSL = "fsl"  # Fixed Sequence Length for training
    UNKNOWN = "unknown"  # Unknown dataset stage


@dataclass
class DatasetConfig:
    """Minimal configuration for OWA datasets with only essential metadata."""

    stage: DatasetStage = DatasetStage.UNKNOWN
    mcap_root_directory: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetConfig":
        return cls(**data)

    @classmethod
    def from_json(cls, path: PathLike) -> "DatasetConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_json(self, path: PathLike) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
