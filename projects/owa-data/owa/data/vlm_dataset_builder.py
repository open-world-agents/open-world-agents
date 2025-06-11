"""
VLM Dataset Builder for OWA data.

This module provides utilities to convert raw OWA event data into formats
suitable for Vision-Language-Action model training, specifically for use
with nanoVLM and similar frameworks.
"""

from typing import Any, Dict, List

from PIL import Image

from owa.env.gst.msg import ScreenEmitted

from .event_encoder import EventEncoder


class VLMDatasetBuilder:
    """
    Simple builder for converting OWA event data to VLA training format.

    This class is a minimal formatter that:
    1. Encodes raw events using EventEncoder
    2. Extracts images from screen events
    3. Formats data for VLA training

    It does NOT generate instructions - instructions must be provided.
    """

    def __init__(self, drop_file_path: bool = True):
        """
        Initialize the VLM dataset builder.

        Args:
            drop_file_path: Whether to drop file_path from encoded events to save tokens.
        """
        self.encoder = EventEncoder(drop_file_path=drop_file_path)

    def process_event_sequence(
        self,
        raw_events: List[Dict[str, Any]],
        instruction: str,
    ) -> Dict[str, Any]:
        """
        Process a sequence of raw events into VLA training format.

        Args:
            raw_events: List of raw event dictionaries from Event Dataset
            instruction: High-level instruction (REQUIRED - what the user wants to do)

        Returns:
            Dict containing:
                - encoded_events: List of encoded event strings (VLA training target)
                - images: List of PIL Images from screen events (VLA training input)
                - instruction: High-level task instruction (VLA training input)
        """
        if not raw_events:
            return {
                "encoded_events": [],
                "images": [],
                "instruction": instruction,
            }

        # Encode all events
        encoded_events = []
        all_images = []

        for event in raw_events:
            try:
                text, images = self.encoder.encode(event)
                encoded_events.append(text)

                # Extract images from screen events
                for image_data in images:
                    if isinstance(image_data, dict) and "screen_event" in image_data:
                        screen_event = image_data["screen_event"]
                        if isinstance(screen_event, ScreenEmitted):
                            # Try to load the image
                            try:
                                image_array = screen_event.lazy_load()
                                if image_array is not None:
                                    # Convert numpy array to PIL Image
                                    # Assuming BGRA format from ScreenEmitted
                                    if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                                        # Convert BGRA to RGB
                                        rgb_array = image_array[:, :, [2, 1, 0]]  # BGR to RGB
                                        pil_image = Image.fromarray(rgb_array, mode="RGB")
                                        all_images.append(pil_image)
                            except Exception as e:
                                # Skip images that can't be loaded
                                print(f"Warning: Could not load image from screen event: {e}")
                                continue

            except Exception as e:
                print(f"Warning: Could not encode event: {e}")
                continue

        return {"encoded_events": encoded_events, "images": all_images, "instruction": instruction}

    def process_batch(
        self,
        event_sequences: List[List[Dict[str, Any]]],
        instructions: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Process multiple event sequences in batch.

        Args:
            event_sequences: List of event sequence lists
            instructions: List of instructions for each sequence (REQUIRED)

        Returns:
            List of processed sequences in VLA format
        """
        if len(event_sequences) != len(instructions):
            raise ValueError(
                f"Number of sequences ({len(event_sequences)}) must match number of instructions ({len(instructions)})"
            )

        results = []
        for events, instruction in zip(event_sequences, instructions):
            processed = self.process_event_sequence(events, instruction)
            results.append(processed)

        return results

    def create_huggingface_dataset(
        self,
        event_sequences: List[List[Dict[str, Any]]],
        instructions: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Create a dataset in HuggingFace format ready for VLA training.

        Args:
            event_sequences: List of event sequence lists
            instructions: List of instructions for each sequence (REQUIRED)

        Returns:
            List of samples in HuggingFace dataset format
        """
        processed_sequences = self.process_batch(event_sequences, instructions)

        # Convert to HuggingFace format
        hf_samples = []
        for sequence in processed_sequences:
            sample = {
                "encoded_events": sequence["encoded_events"],
                "images": sequence["images"],
                "instruction": sequence["instruction"],
            }
            hf_samples.append(sample)

        return hf_samples
