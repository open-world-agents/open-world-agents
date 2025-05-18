from pathlib import Path
from typing import Generator

import numpy as np
from datasets import Dataset
from loguru import logger
from tqdm import tqdm

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
        min_time = start - int(PERCEPTION_SPEC_DICT.start_time * TimeUnits.SECOND)
        max_time = end - int(PERCEPTION_SPEC_DICT.end_time * TimeUnits.SECOND)
        # Generate timestamps in the range [min_time, max_time]
        current_time = min_time
        while current_time < max_time:
            yield current_time
            current_time += int(sample_interval() * TimeUnits.SECOND)


def _generate_dataset(dataset_path: Path) -> Generator[dict, None, None]:
    mcap_files = Path(dataset_path).rglob("*.mcap")
    for file_path in mcap_files:
        with OWAMcapReader(file_path) as reader:
            valid_intervals = [(reader.start_time, reader.end_time)]  # Example intervals
            valid_intervals = [((reader.start_time + reader.end_time) // 2, reader.end_time)]  # Example intervals
            # In real implementation, these intervals are derived by various logics.
            # e.g. representing valid interval with special key, ...
        for now in iter_timestamps(valid_intervals):
            yield {"file_path": file_path.as_posix(), "timestamp": now}


def create_dataset(dataset_path: Path) -> Dataset:
    """
    Create a dataset from the given directory containing .mcap files.
    """
    dataset = Dataset.from_generator(_generate_dataset, gen_kwargs={"dataset_path": dataset_path})
    return dataset


def generate_conversation(dataset: Dataset) -> Dataset:
    """Generate and append new column, 'conversation', to the dataset."""

    def add_conversation(examples):
        conversations = []
        for file_path, now in zip(examples["file_path"], examples["timestamp"]):
            with OWAMcapPerceptionReader(file_path) as reader:
                current_perception = reader.sample(now, spec=PERCEPTION_SPEC_DICT)
                perception_history, conversation = perception_to_conversation(
                    Perception(), current_perception, now=now, spec=PERCEPTION_SPEC_DICT, is_training=True
                )
                conversations.append(conversation)
        examples["conversation"] = conversations
        return examples

    # dataset = dataset.map(add_conversation, num_proc=16, batched=True, desc="Generating conversations")
    dataset.set_transform(add_conversation)
    return dataset
