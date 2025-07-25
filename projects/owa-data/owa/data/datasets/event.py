"""Event dataset implementation."""

from typing import Optional

from datasets import Dataset as HFDataset
from datasets.utils.typing import PathLike

from .base import OWADatasetBase
from .config import EventDatasetConfig


class EventDataset(OWADatasetBase):
    """
    Event dataset with integrated transform functionality.

    This dataset stores raw MCAP events and can apply transforms on-the-fly
    to decode messages and load images.
    """

    owa_config: Optional[EventDatasetConfig]  # type: ignore[override]

    def __init__(self, *args, owa_config: Optional[EventDatasetConfig] = None, **kwargs):
        super().__init__(*args, owa_config=owa_config, **kwargs)

    def set_transform_for_training(self, encoder_type: Optional[str] = None, load_images: Optional[bool] = None):
        """
        Set transform for training/inference that converts raw MCAP events to processed format.

        Transform Output Format:
        - 'encoded_event': str - Human-readable text representation of the event
        - 'image': PIL.Image or None - Loaded image for screen events (if load_images=True)

        Original fields are preserved unless they conflict with transform outputs.

        Args:
            encoder_type: Type of encoder ('hierarchical', 'json', 'flat'). Uses config default if None.
            load_images: Whether to load PIL images for screen events. Uses config default if None.
            encode_actions: Whether to encode action events (keyboard/mouse).

        Example:
            ```python
            dataset = EventDataset(data, owa_config=config)
            dataset.set_transform_for_training(load_images=True, encoder_type="hierarchical")

            # Now accessing dataset items returns:
            # {
            #   'episode_path': 'ep1.mcap',
            #   'topic': 'screen',
            #   'timestamp_ns': 1000000000,
            #   'message_type': 'ScreenCaptured',
            #   'mcap_message': b'...',  # Original data preserved
            #   'encoded_event': 'Screen captured at (100, 200)',  # Added by transform
            #   'image': <PIL.Image>  # Added by transform
            # }
            ```
        """
        if self.owa_config:
            encoder_type = encoder_type or self.owa_config.encoder_type
            load_images = load_images if load_images is not None else self.owa_config.load_images
        else:
            encoder_type = encoder_type or "hierarchical"
            load_images = load_images if load_images is not None else True

        def transform_fn(examples):
            # Lazy import to avoid circular dependencies
            from mcap_owa.highlevel import McapMessage
            from owa.data.encoders import create_encoder

            encoder = create_encoder(encoder_type)
            is_batch = isinstance(examples.get("episode_path", ""), list)

            if is_batch:
                batch_size = len(examples["episode_path"])
                # Preserve all original fields
                result = {key: values for key, values in examples.items()}
                # Add transform outputs
                result["encoded_event"] = [None] * batch_size
                result["image"] = [None] * batch_size

                for i in range(batch_size):
                    single_example = {key: values[i] for key, values in examples.items()}
                    transformed = self._transform_single(single_example, encoder, load_images, McapMessage)
                    result["encoded_event"][i] = transformed["encoded_event"]
                    result["image"][i] = transformed["image"]
                return result
            else:
                # Preserve all original fields
                result = {key: value for key, value in examples.items()}
                # Add transform outputs
                transformed = self._transform_single(examples, encoder, load_images, McapMessage)
                result.update(transformed)
                return result

        self.set_transform(transform_fn)

    def _transform_single(self, example, encoder, load_images, McapMessage):
        """Transform a single example."""
        result = {"encoded_event": None, "image": None}
        try:
            mcap_msg = McapMessage.model_validate_json(example["mcap_message"].decode("utf-8"))
            encoded_text, screen_captured = encoder.encode(mcap_msg)

            if example["topic"] == "screen" and screen_captured and load_images:
                # Simple image loading without path resolution for now
                result["image"] = (
                    screen_captured[0].to_pil_image() if hasattr(screen_captured[0], "to_pil_image") else None
                )

            result["encoded_event"] = encoded_text
        except Exception as e:
            print(f"Warning: Could not process {example['topic']} event: {e}")

        return result

    @staticmethod
    def load_from_disk(dataset_path: PathLike, storage_options: Optional[dict] = None, **kwargs) -> "EventDataset":
        """Load EventDataset from disk with remote filesystem support."""
        from .discovery import resolve_dataset_path_and_config

        # Load HF dataset with remote support
        hf_kwargs = kwargs.copy()
        if storage_options:
            hf_kwargs["storage_options"] = storage_options

        hf_dataset = HFDataset.load_from_disk(dataset_path, **hf_kwargs)

        # Try to load OWA config with remote support
        _, config_data, _ = resolve_dataset_path_and_config(dataset_path, storage_options)
        owa_config = None
        if config_data:
            try:
                owa_config = EventDatasetConfig.from_dict(config_data)
            except Exception:
                pass

        return EventDataset(
            arrow_table=hf_dataset.data,
            info=hf_dataset.info,
            split=hf_dataset.split,
            indices_table=getattr(hf_dataset, "_indices", None),
            fingerprint=getattr(hf_dataset, "_fingerprint", None),
            owa_config=owa_config if isinstance(owa_config, EventDatasetConfig) else None,
        )
