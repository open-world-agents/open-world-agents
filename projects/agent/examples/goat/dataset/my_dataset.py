from agent_system.core.pipe import Pipe
from agent_system.perception.provider import OWAMcapPerceptionReader
from agent_system.pipeline.processors import apply_processor, lazy_load_images, perception_to_conversation
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __getitem__(self, idx):
        # 1. Load query
        file_path, timestamp = self._data[idx]
        # 2. Load perception from file
        current_perception = OWAMcapPerceptionReader(file_path).sample(now=timestamp)
        # 3. Create pending thought
        pending_thought = (
            Pipe([], current_perception, now=timestamp) | perception_to_conversation | lazy_load_images
        ).execute()
        # Note that apply_processor is done in collate_fn.
        return pending_thought


def collate_fn(processor, batch):
    batch = apply_processor(processor, batch)
    return batch
