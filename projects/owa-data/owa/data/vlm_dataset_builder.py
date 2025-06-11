"""
VLM Dataset Builder for OWA data.

This module provides utilities to convert raw OWA event data into formats
suitable for Vision-Language-Action model training, specifically for use
with nanoVLM and similar frameworks.
"""

import json
from typing import Any, Dict, List, Optional

from PIL import Image

from owa.env.gst.msg import ScreenEmitted

from .event_encoder import EventEncoder


class VLMDatasetBuilder:
    """
    Builder for converting OWA event data to VLA training format.

    This class processes raw event datasets and creates structured data
    suitable for training Vision-Language-Action models where:
    - Input: Screen observation + high-level instruction
    - Output: Encoded action events (keyboard/mouse/etc.)
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
        instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a sequence of raw events into VLA training format.

        For VLA training:
        - Input (User): Screen images + high-level instruction ("Please type 'A'")
        - Output (Assistant): Encoded action events (keyboard/mouse events)

        Args:
            raw_events: List of raw event dictionaries from Event Dataset
            instruction: Optional high-level instruction (what to do)

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
                "instruction": instruction or "Perform the demonstrated actions.",
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

        # Generate default instruction if not provided
        if instruction is None:
            instruction = self._generate_default_instruction(raw_events)

        return {"encoded_events": encoded_events, "images": all_images, "instruction": instruction}

    def process_batch(
        self,
        event_sequences: List[List[Dict[str, Any]]],
        instructions: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple event sequences in batch.

        Args:
            event_sequences: List of event sequence lists
            instructions: Optional list of instructions for each sequence

        Returns:
            List of processed sequences in VLA format
        """
        results = []

        for i, events in enumerate(event_sequences):
            instruction = instructions[i] if instructions and i < len(instructions) else None
            processed = self.process_event_sequence(events, instruction)
            results.append(processed)

        return results

    def _generate_default_instruction(self, raw_events: List[Dict[str, Any]]) -> str:
        """Generate a default VLA instruction based on event types."""
        if not raw_events:
            return "Perform the demonstrated actions."

        # Analyze events to generate high-level instruction
        actions = []

        for event in raw_events:
            topic = event.get("topic", "")
            msg_bytes = event.get("msg", b"")

            try:
                msg_data = json.loads(msg_bytes.decode("utf-8"))

                if topic == "keyboard":
                    event_type = msg_data.get("event_type", "")
                    vk = msg_data.get("vk", "")
                    if event_type == "press":
                        # Convert virtual key codes to readable instructions
                        if vk == 65:  # 'A' key
                            actions.append("type 'A'")
                        elif vk == 13:  # Enter key
                            actions.append("press Enter")
                        elif vk == 32:  # Space key
                            actions.append("press Space")
                        else:
                            actions.append(f"press key {vk}")

                elif topic == "mouse":
                    event_type = msg_data.get("event_type", "")
                    x = msg_data.get("x", "")
                    y = msg_data.get("y", "")
                    if event_type == "click":
                        button = msg_data.get("button", "left")
                        actions.append(f"{button} click at ({x}, {y})")
                    elif event_type == "move":
                        actions.append(f"move mouse to ({x}, {y})")
                    elif event_type == "scroll":
                        actions.append("scroll")

            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

        if not actions:
            return "Perform the demonstrated actions."

        # Create high-level instruction from actions
        if len(actions) == 1:
            return f"Please {actions[0]}."
        elif len(actions) <= 3:
            return f"Please {', then '.join(actions)}."
        else:
            return f"Please {actions[0]}, then {actions[1]}, and continue the sequence."

    def create_huggingface_dataset(
        self,
        event_sequences: List[List[Dict[str, Any]]],
        instructions: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Create a dataset in HuggingFace format ready for VLA training.

        Args:
            event_sequences: List of event sequence lists
            instructions: Optional list of instructions

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
