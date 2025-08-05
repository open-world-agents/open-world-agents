"""Clean and minimal OWA Dataset classes."""

import json
import posixpath
from pathlib import Path
from typing import Optional, Union

import fsspec
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from datasets import config
from datasets.utils.file_utils import url_to_fs
from datasets.utils.typing import PathLike

from .config import DatasetConfig, DatasetStage
from .filesystem import resolve_dataset_path_and_config
from .transforms import create_transform


class OWADatasetMixin:
    """Mixin class for OWA Dataset and DatasetDict."""

    _owa_config: DatasetConfig

    @property
    def owa_config(self) -> DatasetConfig:
        """Get the current dataset config."""
        return self._owa_config

    @owa_config.setter
    def owa_config(self, value: DatasetConfig):
        """Set the current dataset config."""
        self._owa_config = value

    @property
    def stage(self) -> DatasetStage:
        """Get the current dataset stage."""
        return self.owa_config.stage

    @stage.setter
    def stage(self, value: DatasetStage):
        """Set the current dataset stage."""
        self.owa_config.stage = value

    @property
    def mcap_root_directory(self) -> Optional[str]:
        """Get the MCAP root directory."""
        return self.owa_config.mcap_root_directory

    @mcap_root_directory.setter
    def mcap_root_directory(self, value: Optional[str]):
        """Set the MCAP root directory."""
        self.owa_config.mcap_root_directory = value

    def auto_set_transform(
        self, stage: Optional[str] = None, mcap_root_directory: Optional[str] = None, **kwargs
    ) -> DatasetStage:
        """Set appropriate transform for a dataset based on its stage."""
        stage = stage or self.stage
        mcap_root_directory = mcap_root_directory or self.mcap_root_directory
        if mcap_root_directory is None:
            raise ValueError("mcap_root_directory must be set")

        self.set_transform(create_transform(stage, mcap_root_directory, **kwargs))
        return stage

    def auto_with_transform(
        self, stage: Optional[str] = None, mcap_root_directory: Optional[str] = None, **kwargs
    ) -> "Dataset | DatasetDict":
        """Set appropriate transform for a dataset based on its stage."""
        stage = stage or self.stage
        mcap_root_directory = mcap_root_directory or self.mcap_root_directory
        if mcap_root_directory is None:
            raise ValueError("mcap_root_directory must be set")

        return self.with_transform(create_transform(stage, mcap_root_directory, **kwargs))


class Dataset(HFDataset, OWADatasetMixin):
    """
    Clean and minimal OWA Dataset class with stage marker.

    This class is a simple wrapper around HuggingFace Dataset that adds:
    - Stage marker (EVENT, BINNED, TOKENIZED, FSL)
    - Config persistence

    All transform logic is handled by separate transform classes.
    """

    def __init__(self, *args, owa_config: DatasetConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self._owa_config = owa_config

    @classmethod
    def from_hf_dataset(cls, hf_dataset: HFDataset, owa_config: DatasetConfig) -> "Dataset":
        format_ = hf_dataset.format
        return cls(
            arrow_table=hf_dataset.data,
            info=hf_dataset.info,
            split=hf_dataset.split,
            indices_table=hf_dataset._indices,
            fingerprint=hf_dataset._fingerprint,
            owa_config=owa_config,
        ).with_format(**format_)

    def save_to_disk(self, dataset_path: PathLike, **kwargs) -> None:  # type: ignore[override]
        super().save_to_disk(dataset_path, **kwargs)
        if self.owa_config is not None:
            config_path = Path(str(dataset_path)) / "owa_config.json"
            self.owa_config.to_json(config_path)

    @staticmethod
    def load_from_disk(dataset_path: PathLike, storage_options: Optional[dict] = None, **kwargs) -> "Dataset":  # type: ignore[override]
        hf_kwargs = kwargs.copy()
        if storage_options:
            hf_kwargs["storage_options"] = storage_options

        # Try to load OWA config with remote support
        _, config_data, _ = resolve_dataset_path_and_config(dataset_path, storage_options)
        owa_config = DatasetConfig(**config_data)

        hf_dataset = HFDataset.load_from_disk(dataset_path, **hf_kwargs)
        return Dataset.from_hf_dataset(hf_dataset, owa_config=owa_config)


class DatasetDict(HFDatasetDict, OWADatasetMixin):
    """
    OWA DatasetDict that inherits from HuggingFace DatasetDict.

    This class extends HFDatasetDict to work with OWA Dataset instances and
    provides automatic OWA config persistence and loading.
    """

    @property
    def owa_config(self) -> Optional[DatasetConfig]:  # type: ignore[override]
        """Get the current dataset config."""
        if not self:
            return None
        return next(iter(self.values())).owa_config

    def save_to_disk(
        self,
        dataset_dict_path: PathLike,
        max_shard_size: Optional[Union[str, int]] = None,
        num_shards: Optional[dict[str, int]] = None,
        num_proc: Optional[int] = None,
        storage_options: Optional[dict] = None,
    ) -> None:
        # Call parent save_to_disk
        super().save_to_disk(dataset_dict_path, max_shard_size, num_shards, num_proc, storage_options)

        # Save OWA config from the first dataset if available
        if self and hasattr(next(iter(self.values())), "owa_config"):
            first_dataset = next(iter(self.values()))
            if first_dataset.owa_config is not None:
                config_path = Path(str(dataset_dict_path)) / "owa_config.json"
                first_dataset.owa_config.to_json(config_path)

    # Copied from https://github.com/huggingface/datasets/blob/main/src/datasets/dataset_dict.py#L1358-L1416
    @staticmethod
    def load_from_disk(
        dataset_dict_path: PathLike,
        keep_in_memory: Optional[bool] = None,
        storage_options: Optional[dict] = None,
    ) -> "DatasetDict":
        """
        Load a dataset that was previously saved using [`save_to_disk`] from a filesystem using `fsspec.spec.AbstractFileSystem`.

        Args:
            dataset_dict_path (`path-like`):
                Path (e.g. `"dataset/train"`) or remote URI (e.g. `"s3//my-bucket/dataset/train"`)
                of the dataset dict directory where the dataset dict will be loaded from.
            keep_in_memory (`bool`, defaults to `None`):
                Whether to copy the dataset in-memory. If `None`, the
                dataset will not be copied in-memory unless explicitly enabled by setting
                `datasets.config.IN_MEMORY_MAX_SIZE` to nonzero. See more details in the
                [improve performance](../cache#improve-performance) section.
            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the file-system backend, if any.

                <Added version="2.8.0"/>

        Returns:
            [`DatasetDict`]

        Example:

        ```py
        >>> ds = load_from_disk('path/to/dataset/directory')
        ```
        """
        fs: fsspec.AbstractFileSystem
        fs, dataset_dict_path = url_to_fs(dataset_dict_path, **(storage_options or {}))

        dataset_dict_json_path = posixpath.join(dataset_dict_path, config.DATASETDICT_JSON_FILENAME)
        dataset_state_json_path = posixpath.join(dataset_dict_path, config.DATASET_STATE_JSON_FILENAME)
        dataset_info_path = posixpath.join(dataset_dict_path, config.DATASET_INFO_FILENAME)
        if not fs.isfile(dataset_dict_json_path):
            if fs.isfile(dataset_info_path) and fs.isfile(dataset_state_json_path):
                raise FileNotFoundError(
                    f"No such file: '{dataset_dict_json_path}'. Expected to load a `DatasetDict` object, but got a `Dataset`. Please use either `datasets.load_from_disk` or `Dataset.load_from_disk` instead."
                )
            raise FileNotFoundError(
                f"No such file: '{dataset_dict_json_path}'. Expected to load a `DatasetDict` object, but provided path is not a `DatasetDict`."
            )

        with fs.open(dataset_dict_json_path, "r", encoding="utf-8") as f:
            splits = json.load(f)["splits"]

        dataset_dict = DatasetDict()
        for k in splits:
            dataset_dict_split_path = posixpath.join(fs.unstrip_protocol(dataset_dict_path), k)
            dataset_dict[k] = Dataset.load_from_disk(
                dataset_dict_split_path,
                keep_in_memory=keep_in_memory,
                storage_options=storage_options,
            )
        return dataset_dict
