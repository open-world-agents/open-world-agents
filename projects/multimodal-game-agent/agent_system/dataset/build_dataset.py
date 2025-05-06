from agent_system.core.pipe import Pipe
from agent_system.perception.provider import OWAMcapPerceptionReader
from agent_system.pipeline.processors import perception_to_conversation

from mcap_owa.highlevel import OWAMcapReader


def iter_timestamps(valid_intervals):
    for start, end in valid_intervals:
        for timestamp in range(start, end, 100):
            yield timestamp


def create_dataset():
    for file_path in ["file1.mcap", "file2.mcap"]:
        with OWAMcapReader(file_path) as reader:
            valid_intervals = [(0, 1000), (2000, 3000)]  # Example intervals
            # In real implementation, these intervals are derived by various logics.
            # e.g. representing valid interval with special key, ...
        for now in iter_timestamps(valid_intervals):
            try:
                current_perception = OWAMcapPerceptionReader(file_path).sample(now)
                perception_history, conversation = perception_to_conversation([], current_perception, now=now)
                yield file_path, now, conversation
            except Exception:
                pass
