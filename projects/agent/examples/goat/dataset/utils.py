from datasets import Dataset as HFDataset
from torch.utils.data import Dataset

from owa.agent.core import OWAMcapPerceptionReader
from owa.agent.core.perception import Perception
from owa.agent.systems.goat import PERCEPTION_SPEC_DICT
from owa.agent.systems.goat.processors import (
    SmolVLMInput,
    apply_processor,
    lazy_load_images,
    perception_to_conversation,
)


class MyDataset(Dataset):
    def __init__(self, base_dataset: HFDataset):
        self._data = base_dataset

    def __getitem__(self, idx):
        item = self._data[idx]

        file_path = item.file_path
        timestamp = item.timestamp
        conversation = getattr(item, "conversation", None)

        if conversation is not None:
            # 1. Load perception from file
            current_perception = OWAMcapPerceptionReader(file_path).sample(now=timestamp, spec=PERCEPTION_SPEC_DICT)
            # 2. Create pending thought
            _, conversation = perception_to_conversation(
                Perception(), current_perception, is_training=True, spec=PERCEPTION_SPEC_DICT
            )

        conversation = lazy_load_images(conversation, mcap_path=file_path)
        # Note that apply_processor is done in collate_fn.
        return conversation


class RepeatingDataset(Dataset):
    def __init__(self, dataset: Dataset, repeat_count: int):
        self.dataset = dataset
        self.repeat_count = repeat_count

    def __len__(self):
        return len(self.dataset) * self.repeat_count

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]


def transform(examples, *, decode_images: bool = True):
    """Transforms which is applied to the HF dataset."""
    # decode the JSON strings
    conversation = [SmolVLMInput.model_validate_json(x) for x in examples["conversation"]]
    # lazy load images
    if decode_images:
        conversation = [lazy_load_images(x, mcap_path=path) for path, x in zip(examples["file_path"], conversation)]
    examples = {
        "images": [x.images for x in conversation],
        "messages": [x.messages for x in conversation],
    }
    return examples


def collate_fn(batch, *, processor):
    batch = [SmolVLMInput.model_validate(x) for x in batch]
    batch = apply_processor(batch, processor=processor, is_training=True)
    return batch
