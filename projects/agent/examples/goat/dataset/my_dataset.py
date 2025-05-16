import functools

from torch.utils.data import Dataset

from owa.agent.core import OWAMcapPerceptionReader, Pipe
from owa.agent.systems.example import PERCEPTION_SPEC_DICT
from owa.agent.systems.example.processors import apply_processor, lazy_load_images, perception_to_conversation


class MyDataset(Dataset):
    def __getitem__(self, idx):
        # 1. Load query
        file_path, timestamp = self._data[idx]
        # 2. Load perception from file
        current_perception = OWAMcapPerceptionReader(file_path).sample(now=timestamp, spec=PERCEPTION_SPEC_DICT)
        # 3. Create pending thought
        pending_thought = (
            Pipe([], current_perception, now=timestamp)
            | functools.partial(perception_to_conversation, spec=PERCEPTION_SPEC_DICT)
            | lazy_load_images
        ).execute()
        # Note that apply_processor is done in collate_fn.
        return pending_thought


def collate_fn(processor, batch):
    batch = apply_processor(processor, batch)
    return batch
