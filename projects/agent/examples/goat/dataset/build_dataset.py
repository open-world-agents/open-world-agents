from mcap_owa.highlevel import OWAMcapReader
from owa.agent.core import OWAMcapPerceptionReader
from owa.agent.systems.example import PERCEPTION_SAMPLING_SPEC
from owa.agent.systems.example.processors import perception_to_conversation


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
                current_perception = OWAMcapPerceptionReader(file_path).sample(now, spec=PERCEPTION_SAMPLING_SPEC)
                perception_history, conversation = perception_to_conversation(
                    [], current_perception, now=now, spec=PERCEPTION_SAMPLING_SPEC
                )
                yield file_path, now, conversation
            except Exception:
                pass
