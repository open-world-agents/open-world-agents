"""Binned dataset implementation."""

from pathlib import Path
from typing import Optional

from datasets import Dataset as HFDataset
from datasets.utils.typing import PathLike

from .base import OWADatasetBase
from .config import BinnedDatasetConfig


class BinnedDataset(OWADatasetBase):
    """
    Binned dataset with integrated transform functionality.

    This dataset stores binned events (state/action sequences) and can apply
    transforms on-the-fly to load images and encode actions.
    """

    def __init__(self, *args, owa_config: Optional[BinnedDatasetConfig] = None, **kwargs):
        super().__init__(*args, owa_config=owa_config, **kwargs)

    def set_transform_for_training(
        self, instruction: str = None, encoder_type: str = None, load_images: bool = None, encode_actions: bool = True
    ):
        """
        Set transform for training/inference that converts binned data to VLA format.

        Transform Output Format:
        - 'instruction': str - Task instruction text
        - 'state': List[PIL.Image] - Loaded images from state sequence (if load_images=True)
        - 'actions': List[str] - Encoded action strings (if encode_actions=True)

        Original fields are preserved unless they conflict with transform outputs.

        Args:
            instruction: Task instruction text. Uses config default if None.
            encoder_type: Type of encoder for actions. Uses config default if None.
            load_images: Whether to load PIL images from state. Uses config default if None.
            encode_actions: Whether to encode action sequences.

        Example:
            ```python
            dataset = BinnedDataset(data, owa_config=config)
            dataset.set_transform_for_training(instruction="Complete the task", load_images=True)

            # Now accessing dataset items returns:
            # {
            #   'episode_path': 'ep1.mcap',
            #   'bin_idx': 0,
            #   'timestamp_ns': 1000000000,
            #   'state': [<PIL.Image>, <PIL.Image>],  # Added by transform
            #   'actions': ['Click at (100, 200)', 'Type "hello"'],  # Added by transform
            #   'instruction': 'Complete the task'  # Added by transform
            # }
            ```
        """
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

            encoder = create_encoder(encoder_type) if encode_actions else None
            is_batch = isinstance(examples.get("episode_path", ""), list)

            if is_batch:
                batch_size = len(examples["episode_path"])
                # Preserve all original fields
                result = {key: values for key, values in examples.items()}
                # Add transform outputs
                result["instruction"] = [instruction] * batch_size
                result["state"] = []
                result["actions"] = []

                for i in range(batch_size):
                    single_example = {key: values[i] for key, values in examples.items()}
                    transformed = self._transform_single(single_example, instruction, encoder, load_images)
                    result["state"].append(transformed["state"])
                    result["actions"].append(transformed["actions"])
                return result
            else:
                # Preserve all original fields
                result = {key: value for key, value in examples.items()}
                # Add transform outputs
                transformed = self._transform_single(examples, instruction, encoder, load_images)
                result.update(transformed)
                return result

        self.set_transform(transform_fn)

    def _transform_single(self, example, instruction, encoder, load_images):
        """Transform a single binned example."""
        result = {"instruction": instruction, "state": [], "actions": []}

        if load_images:
            # Simplified state loading for now - placeholder
            state_sequence = example.get("state", [])
            result["state"] = [f"state_image_{i}" for i in range(len(state_sequence))]  # Placeholder

        if encoder:
            # Simplified action encoding for now - placeholder
            actions_sequence = example.get("actions", [])
            result["actions"] = [f"action_{i}" for i in range(len(actions_sequence))]  # Placeholder

        return result

    @staticmethod
    def load_from_disk(dataset_path: PathLike, storage_options: Optional[dict] = None, **kwargs) -> "BinnedDataset":
        """Load BinnedDataset from disk with remote filesystem support."""
        from .discovery import resolve_dataset_path_and_config

        # Load HF dataset with remote support
        hf_kwargs = kwargs.copy()
        if storage_options:
            hf_kwargs["storage_options"] = storage_options

        hf_dataset = HFDataset.load_from_disk(dataset_path, **hf_kwargs)

        # Try to load OWA config with remote support
        resolved_path, config_data, fs = resolve_dataset_path_and_config(dataset_path, storage_options)
        owa_config = None
        if config_data:
            try:
                owa_config = BinnedDatasetConfig.from_dict(config_data)
            except Exception:
                pass

        return BinnedDataset(
            arrow_table=hf_dataset.data,
            info=hf_dataset.info,
            split=hf_dataset.split,
            indices_table=getattr(hf_dataset, "_indices", None),
            fingerprint=getattr(hf_dataset, "_fingerprint", None),
            owa_config=owa_config if isinstance(owa_config, BinnedDatasetConfig) else None,
        )
