"""Clean, minimal OWA Dataset implementation with integrated transforms."""

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, Optional, Union

from datasets import Dataset as HFDataset
from datasets.utils.typing import PathLike


class DatasetType(StrEnum):
    EVENT = "event"
    BINNED = "binned"
    FSL = "fsl"


@dataclass
class OWADatasetConfig:
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
    dataset_type: DatasetType = DatasetType.FSL
    pad_token_id: int = 0
    max_sequence_length: int = 8192
    load_images: bool = True


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

    def save_to_disk(self, dataset_path: PathLike, **kwargs) -> None:
        super().save_to_disk(dataset_path, **kwargs)
        if self._owa_config is not None:
            config_path = Path(str(dataset_path)) / "owa_config.json"
            self._owa_config.to_json(config_path)

    def apply_transform(self, **transform_kwargs):
        """Apply dataset-specific transform. Override in subclasses."""
        pass


class EventDataset(OWADatasetBase):
    """Event dataset with integrated transform functionality."""

    def __init__(self, *args, owa_config: Optional[EventDatasetConfig] = None, **kwargs):
        super().__init__(*args, owa_config=owa_config, **kwargs)

    def apply_transform(self, encoder_type: str = None, load_images: bool = None, encode_actions: bool = True):
        """Apply event dataset transform using config defaults."""
        if self.owa_config:
            encoder_type = encoder_type or self.owa_config.encoder_type
            load_images = load_images if load_images is not None else self.owa_config.load_images
        else:
            encoder_type = encoder_type or "hierarchical"
            load_images = load_images if load_images is not None else True

        def transform_fn(examples):
            # Lazy import to avoid circular dependencies
            from owa.data.encoders import create_encoder
            from mcap_owa.highlevel import McapMessage
            
            encoder = create_encoder(encoder_type)
            is_batch = isinstance(examples.get("episode_path", ""), list)
            
            if is_batch:
                batch_size = len(examples["episode_path"])
                result = {"encoded_event": [None] * batch_size, "image": [None] * batch_size}
                for i in range(batch_size):
                    single_example = {key: values[i] for key, values in examples.items()}
                    transformed = self._transform_single(single_example, encoder, load_images, McapMessage)
                    result["encoded_event"][i] = transformed["encoded_event"]
                    result["image"][i] = transformed["image"]
                return result
            else:
                return self._transform_single(examples, encoder, load_images, McapMessage)

        self.set_transform(transform_fn)

    def _transform_single(self, example, encoder, load_images, McapMessage):
        """Transform a single example."""
        result = {"encoded_event": None, "image": None}
        try:
            mcap_msg = McapMessage.model_validate_json(example["mcap_message"].decode("utf-8"))
            encoded_text, screen_captured = encoder.encode(mcap_msg)
            
            if example["topic"] == "screen" and screen_captured and load_images:
                # Simple image loading without path resolution for now
                result["image"] = screen_captured[0].to_pil_image() if hasattr(screen_captured[0], 'to_pil_image') else None
            
            result["encoded_event"] = encoded_text
        except Exception as e:
            print(f"Warning: Could not process {example['topic']} event: {e}")
        
        return result

    @staticmethod
    def load_from_disk(dataset_path: PathLike, **kwargs) -> "EventDataset":
        hf_dataset = HFDataset.load_from_disk(dataset_path, **kwargs)
        config_path = Path(str(dataset_path)) / "owa_config.json"
        owa_config = None
        if config_path.exists():
            owa_config = EventDatasetConfig.from_json(config_path)
        return EventDataset(
            arrow_table=hf_dataset.data,
            info=hf_dataset.info,
            split=hf_dataset.split,
            indices_table=getattr(hf_dataset, '_indices', None),
            fingerprint=getattr(hf_dataset, '_fingerprint', None),
            owa_config=owa_config if isinstance(owa_config, EventDatasetConfig) else None,
        )


class BinnedDataset(OWADatasetBase):
    """Binned dataset with integrated transform functionality."""

    def __init__(self, *args, owa_config: Optional[BinnedDatasetConfig] = None, **kwargs):
        super().__init__(*args, owa_config=owa_config, **kwargs)

    def apply_transform(self, instruction: str = None, encoder_type: str = None, load_images: bool = None, encode_actions: bool = True):
        """Apply binned dataset transform using config defaults."""
        if self.owa_config:
            instruction = instruction or self.owa_config.instruction
            encoder_type = encoder_type or self.owa_config.encoder_type
            load_images = load_images if load_images is not None else self.owa_config.load_images
        else:
            instruction = instruction or "Complete the computer task"
            encoder_type = encoder_type or "hierarchical"
            load_images = load_images if load_images is not None else True

        def transform_fn(examples):
            # Lazy import to avoid circular dependencies
            from owa.data.encoders import create_encoder
            
            encoder = create_encoder(encoder_type)
            is_batch = isinstance(examples.get("episode_path", ""), list)
            
            if is_batch:
                batch_size = len(examples["episode_path"])
                result = {"instruction": [instruction] * batch_size, "state": [], "actions": []}
                for i in range(batch_size):
                    single_example = {key: values[i] for key, values in examples.items()}
                    transformed = self._transform_single(single_example, instruction, encoder, load_images)
                    result["state"].append(transformed["state"])
                    result["actions"].append(transformed["actions"])
                return result
            else:
                return self._transform_single(examples, instruction, encoder, load_images)

        self.set_transform(transform_fn)

    def _transform_single(self, example, instruction, encoder, load_images):
        """Transform a single binned example."""
        result = {"instruction": instruction, "state": [], "actions": []}
        
        if load_images:
            # Simplified state loading
            state_sequence = example.get("state", [])
            result["state"] = [f"state_{i}" for i in range(len(state_sequence))]  # Placeholder
        
        if encoder:
            # Simplified action encoding
            actions_sequence = example.get("actions", [])
            result["actions"] = [f"action_{i}" for i in range(len(actions_sequence))]  # Placeholder
        
        return result

    @staticmethod
    def load_from_disk(dataset_path: PathLike, **kwargs) -> "BinnedDataset":
        hf_dataset = HFDataset.load_from_disk(dataset_path, **kwargs)
        config_path = Path(str(dataset_path)) / "owa_config.json"
        owa_config = None
        if config_path.exists():
            owa_config = BinnedDatasetConfig.from_json(config_path)
        return BinnedDataset(
            arrow_table=hf_dataset.data,
            info=hf_dataset.info,
            split=hf_dataset.split,
            indices_table=getattr(hf_dataset, '_indices', None),
            fingerprint=getattr(hf_dataset, '_fingerprint', None),
            owa_config=owa_config if isinstance(owa_config, BinnedDatasetConfig) else None,
        )


class FSLDataset(OWADatasetBase):
    """FSL Dataset that inherits from HuggingFace Dataset."""

    def __init__(self, dataset: HFDataset, image_processor=None, owa_config: Optional[FSLDatasetConfig] = None, **kwargs):
        # Extract data from input dataset
        if hasattr(dataset, 'data'):
            arrow_table = dataset.data
            info = dataset.info
            split = dataset.split
            indices_table = getattr(dataset, '_indices', None)
            fingerprint = getattr(dataset, '_fingerprint', None)
        else:
            arrow_table = dataset
            info = None
            split = None
            indices_table = None
            fingerprint = None
        
        super().__init__(
            arrow_table=arrow_table,
            info=info,
            split=split,
            indices_table=indices_table,
            fingerprint=fingerprint,
            owa_config=owa_config or FSLDatasetConfig("/tmp", DatasetType.FSL),
        )
        
        self.image_processor = image_processor
        self._prepared = False

    def prepare(self):
        """Prepare dataset for sequence learning."""
        try:
            import numpy as np
            self._cumsum = np.cumsum(self["total_token_count"])
            self._prepared = True
        except ImportError:
            raise ImportError("FSLDataset requires numpy. Please install: pip install numpy")

    def get_sequence(self, idx: int) -> Dict[str, Any]:
        """Get a sequence for training."""
        if not self._prepared:
            raise RuntimeError("Dataset must be prepared before use. Call prepare() first.")
        
        # Simplified sequence generation
        return {
            "input_ids": [1, 2, 3, 4],  # Placeholder
            "attention_mask": [1, 1, 1, 1],  # Placeholder
            "images": [],  # Placeholder
        }


# Factory functions
def create_owa_dataset(dataset_type: DatasetType, *args, owa_config: Optional[OWADatasetConfig] = None, **kwargs) -> OWADatasetBase:
    if dataset_type == DatasetType.EVENT:
        config = owa_config if isinstance(owa_config, EventDatasetConfig) else None
        return EventDataset(*args, owa_config=config, **kwargs)
    elif dataset_type == DatasetType.BINNED:
        config = owa_config if isinstance(owa_config, BinnedDatasetConfig) else None
        return BinnedDataset(*args, owa_config=config, **kwargs)
    elif dataset_type == DatasetType.FSL:
        config = owa_config if isinstance(owa_config, FSLDatasetConfig) else None
        return FSLDataset(*args, owa_config=config, **kwargs)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def load_owa_dataset(dataset_path: PathLike, **kwargs) -> OWADatasetBase:
    config_path = Path(str(dataset_path)) / "owa_config.json"
    if config_path.exists():
        try:
            config_data = json.loads(config_path.read_text())
            dataset_type = DatasetType(config_data.get("dataset_type"))
            if dataset_type == DatasetType.EVENT:
                return EventDataset.load_from_disk(dataset_path, **kwargs)
            elif dataset_type == DatasetType.BINNED:
                return BinnedDataset.load_from_disk(dataset_path, **kwargs)
        except Exception:
            pass
    
    # Fallback to regular HF dataset
    hf_dataset = HFDataset.load_from_disk(dataset_path, **kwargs)
    return OWADatasetBase(
        arrow_table=hf_dataset.data,
        info=hf_dataset.info,
        split=hf_dataset.split,
        indices_table=getattr(hf_dataset, '_indices', None),
        fingerprint=getattr(hf_dataset, '_fingerprint', None),
    )


def convert_hf_dataset_to_owa(hf_dataset: HFDataset, dataset_type: DatasetType, owa_config: Optional[OWADatasetConfig] = None) -> OWADatasetBase:
    return create_owa_dataset(
        dataset_type,
        arrow_table=hf_dataset.data,
        info=hf_dataset.info,
        split=hf_dataset.split,
        indices_table=getattr(hf_dataset, '_indices', None),
        fingerprint=getattr(hf_dataset, '_fingerprint', None),
        owa_config=owa_config,
    )
