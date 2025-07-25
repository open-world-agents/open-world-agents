"""Base dataset class for OWA datasets."""

from pathlib import Path
from typing import Optional

from datasets import Dataset as HFDataset
from datasets.utils.typing import PathLike

from .config import OWADatasetConfig
from .discovery import resolve_dataset_path_and_config


class OWADatasetBase(HFDataset):
    """Base class for OWA datasets with config persistence."""

    def __init__(self, *args, owa_config: Optional[OWADatasetConfig] = None, **kwargs):
        self._owa_config = owa_config
        super().__init__(*args, **kwargs)

    @property
    def owa_config(self) -> Optional[OWADatasetConfig]:
        return self._owa_config

    @owa_config.setter
    def owa_config(self, config: OWADatasetConfig) -> None:
        self._owa_config = config

    def save_to_disk(self, dataset_path: PathLike, **kwargs) -> None:  # type: ignore[override]
        """Save dataset to disk with automatic config persistence."""
        super().save_to_disk(dataset_path, **kwargs)
        if self._owa_config is not None:
            config_path = Path(str(dataset_path)) / "owa_config.json"
            self._owa_config.to_json(config_path)

    def set_transform(self, transform, columns=None, output_all_columns=True):
        """
        Set a transform function to be applied on-the-fly to the dataset.

        The transform function is applied to examples when they are accessed,
        allowing for dynamic data processing without modifying the underlying data.

        Args:
            transform: Function that takes a batch of examples and returns transformed examples
            columns: Columns to apply the transform to (None = all columns)
            output_all_columns: Whether to output all columns or only transformed ones
        """
        super().set_transform(transform, columns, output_all_columns)

    @staticmethod
    def load_from_disk(dataset_path: PathLike, storage_options: Optional[dict] = None, **kwargs) -> "OWADatasetBase":  # type: ignore[override]
        """
        Load OWA dataset from disk with remote filesystem support.

        Args:
            dataset_path: Path to dataset directory (local or remote)
            storage_options: Options for remote filesystem access
            **kwargs: Additional arguments for HF Dataset loading

        Returns:
            OWADatasetBase instance with config loaded
        """
        # Load HF dataset with remote support
        hf_kwargs = kwargs.copy()
        if storage_options:
            hf_kwargs["storage_options"] = storage_options

        hf_dataset = HFDataset.load_from_disk(dataset_path, **hf_kwargs)

        # Try to load OWA config
        _, config_data, _ = resolve_dataset_path_and_config(dataset_path, storage_options)
        owa_config = None
        if config_data:
            try:
                owa_config = OWADatasetConfig.from_dict(config_data)
            except Exception:
                pass

        return OWADatasetBase(
            arrow_table=hf_dataset.data,
            info=hf_dataset.info,
            split=hf_dataset.split,
            indices_table=getattr(hf_dataset, "_indices", None),
            fingerprint=getattr(hf_dataset, "_fingerprint", None),
            owa_config=owa_config,
        )
