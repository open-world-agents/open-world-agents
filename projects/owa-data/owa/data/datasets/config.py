"""Dataset configuration classes."""

from enum import StrEnum
from typing import Optional

from datasets.utils.typing import PathLike
from pydantic import BaseModel

from owa.core.utils import EasyDict


class DatasetStage(StrEnum):
    """Dataset processing stages in the OWA pipeline."""

    EVENT = "event"  # Raw MCAP events
    BINNED = "binned"  # Time-binned events (state/action sequences)
    TOKENIZED = "tokenized"  # Tokenized events from EpisodeTokenizer
    FSL = "fsl"  # Fixed Sequence Length for training
    UNKNOWN = "unknown"  # Unknown dataset stage


class DatasetConfig(BaseModel):
    """Configuration for OWA datasets with predefined common fields."""

    # Core fields
    stage: DatasetStage = DatasetStage.UNKNOWN
    mcap_root_directory: Optional[str] = None

    # Common configuration fields
    mcap_to_event_config: Optional[EasyDict] = None
    event_to_fsl_config: Optional[EasyDict] = None

    def to_json(self, path: PathLike) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=4))
