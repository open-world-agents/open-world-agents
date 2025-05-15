from pathlib import Path

import numpy as np

from mcap_owa.highlevel import OWAMcapReader
from owa.agent.core import OWAMcapPerceptionReader
from owa.agent.systems.example import PERCEPTION_SAMPLING_SPEC
from owa.agent.systems.example.processors import perception_to_conversation
from owa.core.time import TimeUnits


def sample_interval():
    # t' = t + uniform[0, PERCEPTION_SAMPLING_SPEC.duration]
    return np.random.rand() * PERCEPTION_SAMPLING_SPEC.duration


def iter_timestamps(valid_intervals):
    for start, end in valid_intervals:
        for timestamp in range(start, end, int(sample_interval() * TimeUnits.SECOND)):
            yield timestamp


def create_dataset(dataset_path: Path):
    mcap_files = Path(dataset_path).rglob("*.mcap")
    for file_path in mcap_files:
        with OWAMcapReader(file_path) as reader:
            valid_intervals = [(reader.start_time, reader.end_time)]  # Example intervals
            # In real implementation, these intervals are derived by various logics.
            # e.g. representing valid interval with special key, ...
        with OWAMcapPerceptionReader(file_path) as reader:
            for now in iter_timestamps(valid_intervals):
                try:
                    current_perception = reader.sample(now, spec=PERCEPTION_SAMPLING_SPEC)
                    perception_history, conversation = perception_to_conversation(
                        [], current_perception, now=now, spec=PERCEPTION_SAMPLING_SPEC
                    )
                    yield file_path, now, conversation
                except Exception:
                    import traceback

                    traceback.print_exc()
                    pass
