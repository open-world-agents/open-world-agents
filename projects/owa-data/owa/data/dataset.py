import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Optional, Union

from datasets import Dataset
from datasets.utils.typing import PathLike


class DatasetType(StrEnum):
    EVENT = "event"
    BINNED = "binned"
    FSL = "fsl"


@dataclass
class OWADatasetConfig:
    mcap_root_directory: PathLike
    dataset_type: DatasetType

    @staticmethod
    def from_json(path: PathLike) -> "OWADatasetConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return OWADatasetConfig(**data)

    def to_json(self, path: PathLike):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)


class EventDataset(Dataset):
    def __init__(self, *args, owa_dataset_config: OWADatasetConfig, **kwargs):
        self._owa_dataset_config = owa_dataset_config
        super().__init__(*args, **kwargs)

    def save_to_disk(
        self,
        dataset_path: PathLike,
        max_shard_size: Optional[Union[str, int]] = None,
        num_shards: Optional[int] = None,
        num_proc: Optional[int] = None,
        storage_options: Optional[dict] = None,
    ):
        super().save_to_disk(dataset_path, max_shard_size, num_shards, num_proc, storage_options)
        # Save the config as well
        self._owa_dataset_config.to_json(Path(dataset_path) / "owa_config.json")

    @staticmethod
    def load_from_disk(
        dataset_path: PathLike,
        keep_in_memory: Optional[bool] = None,
        storage_options: Optional[dict] = None,
    ) -> "EventDataset":
        # Load the config
        config_path = Path(dataset_path) / "owa_config.json"
        if not config_path.exists():
            raise ValueError(f"OWA config not found at {config_path}")
        config = OWADatasetConfig.from_json(config_path)
        # Load the dataset
        ds = super(EventDataset, EventDataset).load_from_disk(dataset_path, keep_in_memory, storage_options)
        return EventDataset(ds.data, owa_dataset_config=config)
