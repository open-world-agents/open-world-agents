from pathlib import Path
from typing import Generator

import numpy as np
from datasets import Dataset, DatasetDict, DatasetInfo
from tqdm import tqdm

from mcap_owa.highlevel import OWAMcapReader
from owa.agent.core import OWAMcapPerceptionReader
from owa.agent.core.perception import Perception
from owa.agent.dataset import Intervals
from owa.agent.dataset.interval_selector import KeyPressIntervalExtractor
from owa.agent.systems.goat import PERCEPTION_SPEC_DICT, EventProcessor
from owa.agent.systems.goat.processors import perception_to_conversation
from owa.core.time import TimeUnits
from owa.env.desktop.constants import VK

RECORD_START_STOP_KEY = VK.F9
RECORD_PAUSE_KEY = VK.F10


def sample_interval():
    # t' = t + uniform[0, PERCEPTION_SAMPLING_SPEC.duration]
    return np.random.rand() * PERCEPTION_SPEC_DICT.duration


def iter_timestamps(valid_intervals: Intervals) -> Generator[int, None, None]:
    for start, end in valid_intervals:
        min_time = start - int(PERCEPTION_SPEC_DICT.start_time * TimeUnits.SECOND)
        max_time = end - int(PERCEPTION_SPEC_DICT.end_time * TimeUnits.SECOND)
        # Generate timestamps in the range [min_time, max_time]
        current_time = min_time
        while current_time < max_time:
            yield current_time
            current_time += int(sample_interval() * TimeUnits.SECOND)


def check_mcap_sanity(file_path: Path) -> bool:
    """Check if the mcap file is valid and contains the required topics."""
    try:
        # NOTE: This is not sufficient to check if the file is valid. Error may be raised when reading the whole file.
        with OWAMcapReader(file_path) as reader:  # noqa: F841
            for topic, timestamp, msg in reader.iter_decoded_messages():
                break  # Just to check if we can read the file
            return True
        # pair video must exists
        assert file_path.with_suffix(".mkv").exists(), f"File {file_path} does not exist"
    except Exception as e:
        print(f"Error reading mcap file {file_path}: {e}")
        return False


# TODO: parallelize mcap file processing
def _generate_train_dataset(mcap_files: list[Path]) -> Generator[dict, None, None]:
    interval_extractor = KeyPressIntervalExtractor()
    for file_path in tqdm(mcap_files, desc="Generating train dataset", unit="file"):
        try:
            valid_intervals = interval_extractor.extract_intervals(file_path)
            # Filter only intervals longer than 30 seconds (SUPER HEXAGON)
            valid_intervals = interval_extractor.filter_by_duration(valid_intervals, 60 * TimeUnits.SECOND)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
        for now in iter_timestamps(valid_intervals):
            yield {"file_path": file_path.as_posix(), "timestamp": now}


def _generate_test_dataset(mcap_files: list[Path]) -> Generator[dict, None, None]:
    interval_extractor = KeyPressIntervalExtractor()
    for file_path in tqdm(mcap_files, desc="Generating train dataset", unit="file"):
        try:
            valid_intervals = interval_extractor.extract_intervals(file_path)
            # Filter only intervals longer than 30 seconds (SUPER HEXAGON)
            valid_intervals = interval_extractor.filter_by_duration(valid_intervals, 30 * TimeUnits.SECOND)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
        for now in iter_timestamps(valid_intervals):
            yield {"file_path": file_path.as_posix(), "timestamp": now}


def create_dataset(train_files: list[Path], test_files: list[Path]) -> DatasetDict:
    """Create a dataset from the given mcap files."""
    # Check if the mcap files are valid
    for file_path in train_files + test_files:
        if not check_mcap_sanity(file_path):
            raise ValueError(f"Invalid mcap file: {file_path}")

    train_dataset = Dataset.from_generator(
        _generate_train_dataset, gen_kwargs={"mcap_files": train_files}, split="train"
    )
    test_dataset = Dataset.from_generator(_generate_test_dataset, gen_kwargs={"mcap_files": test_files}, split="test")

    info_to_update = DatasetInfo(
        description="",
        dataset_name="open-world-agents/goat",
        homepage="https://github.com/open-world-agents",
    )
    train_dataset.info.update(info_to_update)
    test_dataset.info.update(info_to_update)

    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
    return dataset_dict


def generate_conversation(dataset: DatasetDict) -> DatasetDict:
    """Generate and append new column, 'conversation', to the dataset."""

    event_processor = EventProcessor()  # TODO: make this configurable

    def add_conversation(examples):
        conversations = []
        for file_path, now in zip(examples["file_path"], examples["timestamp"]):
            with OWAMcapPerceptionReader(file_path) as reader:
                current_perception = reader.sample(now, spec=PERCEPTION_SPEC_DICT)
                perception_history, conversation = perception_to_conversation(
                    Perception(),
                    current_perception,
                    now=now,
                    spec=PERCEPTION_SPEC_DICT,
                    is_training=True,
                    event_processor=event_processor,
                )
                conversations.append(conversation.model_dump_json(exclude_none=True))
        examples["conversation"] = conversations
        return examples

    dataset = dataset.map(add_conversation, num_proc=112, batched=True, batch_size=64, desc="Generating conversations")
    # dataset.set_transform(add_conversation)
    return dataset


def generate_sampling_weight(dataset: DatasetDict) -> DatasetDict:
    """Generate and append new column, 'sampling_weight', to the dataset."""

    def add_sampling_weight(examples):
        examples["sampling_weight"] = [1.0] * len(examples["file_path"])
        return examples

    dataset = dataset.map(add_sampling_weight, num_proc=112, batched=True, desc="Generating sampling weights")
    return dataset
