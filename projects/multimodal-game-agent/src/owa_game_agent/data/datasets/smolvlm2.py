from pathlib import Path
from typing import Any, List

import albumentations as A
import cv2
import line_profiler
import numpy as np
import torch
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast
from transformers.models.smolvlm import SmolVLMProcessor

from owa.env.gst.msg import ScreenEmitted

from ..query import OWAMcapQuery
from ..sample import OWATrainingSample
from ..sample_processor import SampleProcessor

CHAT_INSTRUCTION = """You are playing Super Hexagon, a fast-paced game that requires precise control and timing. The current keyboard state, which represents the keys that are pressed, is {keyboard_state}.
After this prompt, you will receive {len_images} sequential image frames that show the game's visual history from the past to the present.
Using the current keyboard state and the image sequence, predict the future sequence of keyboard actions. For each action, include the timestamp when it should be executed."""


class SmolVLMInput(BaseModel):
    images: list[Any]
    messages: list[dict]


def pad_to_square(image: Image.Image) -> Image.Image:
    """
    Pad an image to make it square. The image will be centered in the square.
    """
    width, height = image.size
    if width == height:
        return image
    elif width > height:
        new_image = Image.new("RGB", (width, width), (255, 255, 255))
        new_image.paste(image, (0, (width - height) // 2))
        return new_image
    else:
        new_image = Image.new("RGB", (height, height), (255, 255, 255))
        new_image.paste(image, ((height - width) // 2, 0))
        return new_image


@line_profiler.profile  # profile: 0.266s
def sample_to_smolvlm_input(sample: OWATrainingSample, rotate_augment=False) -> SmolVLMInput:
    """
    Convert an OWATrainingSample to a SmolvlmInput object.

    Args:
        sample: The training sample to convert
    """

    # Convert state_keyboard to string representation
    keyboard_state = "".join(sample.state_keyboard)

    # Convert action_keyboard to string representation
    keyboard_action = "".join(str(item) for pair in sample.action_keyboard for item in pair)

    # Convert screen states to PIL images
    # profile: 97.8% (0.261s)
    state_screen = []
    for timestamp, screen in sample.state_screen:
        # FIXME: more graceful logic
        if isinstance(screen, np.ndarray):
            # convert BGRA to RGB PIL image
            screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2RGB)
            state_screen.append(Image.fromarray(screen))
        else:
            state_screen.append(ScreenEmitted(**screen).to_pil_image())

    # SmolVLM2 only takes a square image, so we will pad the images to make them square
    state_screen = [pad_to_square(image) for image in state_screen]

    if rotate_augment:
        # Rotate randomly with center (256, 256)
        additional_targets = {}
        for i in range(1, len(state_screen)):
            additional_targets[f"image_{i}"] = "image"

        transform = A.Compose(
            [
                A.Rotate(limit=(-180, 180), p=1.0),
                A.CenterCrop(256, 256, p=1.0),
                A.Resize(512, 512, p=1.0),
            ],
            additional_targets=additional_targets,
        )
        state_screen = [np.asarray(image) for image in state_screen]
        state_screen = transform(
            image=state_screen[0], **{f"image_{i}": image for i, image in enumerate(state_screen[1:], start=1)}
        )
        state_screen = [state_screen["image"]] + [state_screen[f"image_{i}"] for i in range(1, len(state_screen))]
        # Convert to PIL image
        state_screen = [Image.fromarray(image) for image in state_screen]

    # # resize to 512x512
    # state_screen = [image.resize((512, 512), Image.BICUBIC) for image in state_screen]
    # # crop center, with size 256x256
    # state_screen = [image.crop((128, 128, 384, 384)) for image in state_screen]

    len_images = len(state_screen)

    # Create messages for SmolVLM
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": CHAT_INSTRUCTION.format(len_images=len_images, keyboard_state=keyboard_state)
                    + "<image>" * len_images,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": keyboard_action},
            ],
        },
    ]

    return SmolVLMInput(images=state_screen, messages=messages)


class SmolVLM2Dataset(Dataset):
    """
    Self-contained dataset for OWA game training.
    Provides direct access to processed text and images ready for training.
    """

    def __init__(self, query_path: str, repeat_n: int = 1):
        """
        Initialize the dataset.

        Args:
            query_path: Path to JSONL file containing queries
            repeat_n: Number of times to repeat the dataset. Useful for efficient loading of small datasets.
        """
        self.query_path = Path(query_path)
        self.queries = self.load_queries()
        self.sample_processor = SampleProcessor()
        self._repeat_n = repeat_n
        logger.info(f"Loaded {len(self.queries)} queries from {query_path}")

    def load_queries(self) -> List[OWAMcapQuery]:
        """Load queries from a JSONL file."""
        queries = []
        with open(self.query_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        queries.append(OWAMcapQuery.model_validate_json(line))
                    except Exception as e:
                        logger.error(f"Error parsing query: {e}")
        return queries

    def __len__(self):
        return len(self.queries) * self._repeat_n

    @line_profiler.profile  # profile: 1.99s
    def __getitem__(self, idx):
        """
        Get a processed item ready for training.

        Returns a dictionary with:
            - text: The processed text for the model (either as chat template or raw messages)
            - images: List of PIL images from the sample
            - sample_id: Original index in the dataset
        """
        # Consider the repeat_n parameter
        idx = idx % len(self.queries)
        query = self.queries[idx]

        sample = query.to_sample()  # profile: 72.5% (0.723s)

        sample = self.sample_processor.tokenize(sample)

        # Use SampleProcessor to process the sample
        vlm_input = sample_to_smolvlm_input(sample)  # profile: 26.8% (0.267s)

        # Extract images and messages
        images = vlm_input.images
        messages = vlm_input.messages

        return {"messages": messages, "images": images, "sample_id": idx}


def collate_fn(examples, processor: SmolVLMProcessor):
    """
    Collate function that works with the self-contained OWAGameDataset.

    Args:
        examples: List of items from OWAGameDataset
        processor: Processor to tokenize text and process images
    """
    texts = []
    images = []
    for ex in examples:
        # strip \n in "<|im_start|>User:...<end_of_utterance>\nAssistant: <end_of_utterance>\n"
        texts.append(processor.apply_chat_template(ex["messages"], tokenize=False).strip())
        images.append(ex["images"])

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # TODO: verify whether special token is handled correctly
    tokenizer: GPT2TokenizerFast = processor.tokenizer

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels: torch.IntTensor = batch["input_ids"].clone()

    # First, mask all tokens (set to -100)
    labels.fill_(-100)

    # Pattern for start of assistant's response
    start_pattern = tokenizer("<end_of_utterance>\nAssistant: ")["input_ids"]
    # End pattern is just the <end_of_utterance> token
    end_token = tokenizer("<end_of_utterance>")["input_ids"][0]

    # Process each sequence in the batch
    for i in range(labels.size(0)):
        input_ids = batch["input_ids"][i].tolist()

        # Find the start pattern
        for j in range(len(input_ids) - len(start_pattern)):
            if input_ids[j : j + len(start_pattern)] == start_pattern:
                start_idx = j + len(start_pattern)

                # Find the next end token after the start pattern
                for k in range(start_idx, len(input_ids)):
                    if input_ids[k] == end_token:
                        end_idx = k + 1  # include <end_of_utterance> token

                        # Unmask the assistant's response (keep these tokens in the loss computation)
                        labels[i, start_idx:end_idx] = batch["input_ids"][i, start_idx:end_idx]
                        break
                break

    batch["labels"] = labels
    return batch
