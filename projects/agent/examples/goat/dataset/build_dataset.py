from pathlib import Path
from typing import Generator

import numpy as np
from datasets import Dataset, DatasetInfo

from mcap_owa.highlevel import OWAMcapReader
from owa.agent.core import OWAMcapPerceptionReader
from owa.agent.core.perception import Perception
from owa.agent.systems.goat import PERCEPTION_SPEC_DICT
from owa.agent.systems.goat.processors import perception_to_conversation
from owa.core.time import TimeUnits
from owa.env.desktop.constants import VK

RECORD_START_STOP_KEY = VK.F9
RECORD_PAUSE_KEY = VK.F10


def sample_interval():
    # t' = t + uniform[0, PERCEPTION_SAMPLING_SPEC.duration]
    return np.random.rand() * PERCEPTION_SPEC_DICT.duration


def iter_timestamps(valid_intervals: list[tuple[int, int]]) -> Generator[int, None, None]:
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
        valid_intervals = []
        with OWAMcapReader(file_path) as reader:
            for topic, timestamp, msg in reader.iter_decoded_messages(topics=["keyboard"]):
                if msg.event_type == "release" and msg.vk == RECORD_START_STOP_KEY:
                    valid_intervals.append(timestamp)
                elif msg.vk == RECORD_PAUSE_KEY:
                    raise NotImplementedError("Pause key is not implemented")
        valid_intervals = list(zip(valid_intervals[::2], valid_intervals[1::2]))
        # Filter only intervals longer than 60 seconds (SUPER HEXAGON)
        valid_intervals = list(filter(lambda x: x[1] - x[0] > 60 * TimeUnits.SECOND, valid_intervals))
        for now in iter_timestamps(valid_intervals):
            yield {"file_path": file_path.as_posix(), "timestamp": now}


def create_dataset(dataset_path: Path) -> Dataset:
    """
    Create a dataset from the given directory containing .mcap files.
    """
    dataset = Dataset.from_generator(_generate_dataset, gen_kwargs={"dataset_path": dataset_path})
    dataset.info.update(
        DatasetInfo(
            description="",
            dataset_name="open-world-agents/goat",
            homepage="https://github.com/open-world-agents",
        )
    )
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
                conversations.append(conversation.model_dump_json(exclude_none=True))
        examples["conversation"] = conversations
        return examples

    dataset = dataset.map(add_conversation, num_proc=16, batched=True, desc="Generating conversations")
    # dataset.set_transform(add_conversation)
    return dataset


def generate_sampling_weight(dataset: Dataset) -> Dataset:
    """Generate and append new column, 'sampling_weight', to the dataset."""

    def add_sampling_weight(examples):
        examples["sampling_weight"] = [1.0] * len(examples["file_path"])
        return examples

    dataset = dataset.map(add_sampling_weight, num_proc=16, batched=True, desc="Generating sampling weights")
    return dataset
