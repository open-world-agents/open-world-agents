from pathlib import Path

import numpy as np

from mcap_owa.highlevel import OWAMcapReader
from owa.agent.core import OWAMcapPerceptionReader
from owa.agent.core.perception import Perception
from owa.agent.systems.goat import PERCEPTION_SPEC_DICT
from owa.agent.systems.goat.processors import perception_to_conversation
from owa.core.time import TimeUnits


def sample_interval():
    # t' = t + uniform[0, PERCEPTION_SAMPLING_SPEC.duration]
    return np.random.rand() * PERCEPTION_SPEC_DICT.duration


def iter_timestamps(valid_intervals):
    for start, end in valid_intervals:
        for timestamp in range(start, end, int(sample_interval() * TimeUnits.SECOND)):
            yield timestamp


def validate_perception(current_perception):
    for event in current_perception:
        ...


def create_dataset(dataset_path: Path):
    mcap_files = Path(dataset_path).rglob("*.mcap")
    for file_path in mcap_files:
        with OWAMcapReader(file_path) as reader:
            valid_intervals = [(reader.start_time, reader.end_time)]  # Example intervals
            valid_intervals = [((reader.start_time + reader.end_time) // 2, reader.end_time)]  # Example intervals
            # In real implementation, these intervals are derived by various logics.
            # e.g. representing valid interval with special key, ...
        with OWAMcapPerceptionReader(file_path) as reader:
            for now in iter_timestamps(valid_intervals):
                try:
                    current_perception = reader.sample(now, spec=PERCEPTION_SPEC_DICT)
                    validate_perception(current_perception)
                    perception_history, conversation = perception_to_conversation(
                        Perception(), current_perception, now=now, spec=PERCEPTION_SPEC_DICT
                    )
                    yield file_path, now, conversation
                except Exception:
                    import traceback

                    traceback.print_exc()
                    pass
