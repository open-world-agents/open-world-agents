"""MCAP to events processing functionality for OWA data pipeline."""

import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional

from tqdm import tqdm

from mcap_owa.highlevel import OWAMcapReader
from owa.data.interval.selector import InactivityFilter
from owa.data.processing.resampler import EventResampler, create_resampler


def process_raw_events_file(
    episode_path: str,
    rate_settings: Dict[str, float],
    keep_topics: Optional[List[str]] = None,
    mcap_root_directory: Optional[str] = None,
) -> List[Dict]:
    """
    Process MCAP file with resampling.

    Args:
        episode_path: Path to the MCAP file to process
        rate_settings: Mapping from topic to desired rate (Hz) for resampling
        keep_topics: Optional list of topics to keep. If None, all topics are kept
        mcap_root_directory: Optional root directory for storing relative paths

    Returns:
        List of event dictionaries containing processed events
    """
    events: List[Dict] = []
    interval_extractor = InactivityFilter()
    valid_intervals = interval_extractor.extract_intervals(Path(episode_path))

    # Initialize resamplers for all topics
    resamplers: Dict[str, EventResampler] = {}
    for topic in keep_topics or []:
        rate_hz = rate_settings.get(topic, 0)  # 0 = no rate limit
        min_interval_ns = 0 if rate_hz == 0 else int((1.0 / rate_hz) * 1e9)
        resamplers[topic] = create_resampler(topic, min_interval_ns=min_interval_ns)

    with OWAMcapReader(Path(episode_path)) as reader:
        for interval in valid_intervals:
            for mcap_msg in reader.iter_messages(start_time=interval.start, end_time=interval.end, topics=keep_topics):
                topic = mcap_msg.topic

                # Process event through resampler
                resamplers[topic].add_event(mcap_msg)
                ready_events = resamplers[topic].pop_event()

                # Process all ready events
                for mcap_message_obj in ready_events:
                    # Serialize McapMessage to bytes using model_dump_json
                    mcap_message_bytes = mcap_message_obj.model_dump_json().encode("utf-8")

                    # Store relative path if mcap_root_directory is provided
                    stored_episode_path = episode_path
                    if mcap_root_directory:
                        stored_episode_path = Path(episode_path).relative_to(mcap_root_directory).as_posix()

                    events.append(
                        {
                            "episode_path": stored_episode_path,
                            "topic": topic,
                            "timestamp_ns": mcap_message_obj.timestamp,
                            "message_type": mcap_message_obj.message_type,
                            "mcap_message": mcap_message_bytes,  # Store serialized bytes
                        }
                    )

    return events


def generate_event_examples(
    episode_paths: List[str],
    rate_settings: Dict[str, float],
    keep_topics: Optional[List[str]] = None,
    num_workers: int = 4,
    mcap_root_directory: Optional[str] = None,
    on_error: Optional[Callable[[str, BaseException], None]] = None,
):
    """
    Generator function that yields event examples by processing each raw events file
    in parallel using multiple processes.

    Args:
        episode_paths: List of MCAP file paths (strings).
        rate_settings: Mapping from topic to desired rate (Hz).
        keep_topics: Optional list of topics to keep. If None, all topics are kept.
        num_workers: Number of parallel worker processes.
        mcap_root_directory: Optional root directory for storing relative paths.
        on_error: Optional callback function to handle errors.

    Yields:
        Individual event dictionaries suitable for Hugging Face Dataset.
    """
    total_files = len(episode_paths)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {
            executor.submit(process_raw_events_file, fp, rate_settings, keep_topics, mcap_root_directory): fp
            for fp in episode_paths
        }
        with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
            for future in as_completed(future_to_path):
                fp = future_to_path[future]
                try:
                    events = future.result()
                    yield from events
                except Exception as e:
                    if on_error:
                        on_error(fp, e)
                    else:
                        warnings.warn(
                            f"Failed to process file {Path(fp).name}: {e}", category=RuntimeWarning, stacklevel=2
                        )
                finally:
                    pbar.update(1)


def create_event_dataset_from_mcaps(
    episode_paths: List[Path],
    rate_settings: Dict[str, float],
    keep_topics: Optional[List[str]] = None,
    num_workers: int = 4,
    split: str = "train",
    mcap_root_directory: Optional[str] = None,
):
    """
    Create a Hugging Face event dataset from the given MCAP file paths by streaming
    examples from a generator.

    Args:
        episode_paths: List of pathlib.Path objects pointing to MCAP files.
        rate_settings: Mapping from topic to rate (Hz) to apply drop-only downsampling.
        keep_topics: Optional list of topics to keep. If None, all topics are kept.
        num_workers: Number of worker processes for parallel file processing.
        split: Dataset split name (default: "train").
        mcap_root_directory: Optional root directory for storing relative paths.

    Returns:
        A Hugging Face Dataset containing the combined events.
    """
    from datasets import Dataset as HFDataset
    from datasets import DatasetInfo as HFDatasetInfo
    from datasets import Features, Value

    from owa.data.datasets import Dataset, DatasetConfig, DatasetStage

    episode_path_strs = [str(fp) for fp in episode_paths]

    features = Features(
        {
            "episode_path": Value("string"),
            "topic": Value("string"),
            "timestamp_ns": Value("int64"),
            "message_type": Value("string"),
            "mcap_message": Value("binary"),  # Use bytes serialization for McapMessage
        }
    )

    # Create HF Dataset first
    hf_dataset = HFDataset.from_generator(
        generate_event_examples,
        gen_kwargs={
            "episode_paths": episode_path_strs,
            "rate_settings": rate_settings,
            "keep_topics": keep_topics,
            "num_workers": num_workers,
            "mcap_root_directory": mcap_root_directory,
        },
        features=features,
        split=split,
    )

    # Update dataset info
    info_to_update = HFDatasetInfo(
        description="",
        dataset_name="open-world-agents/goat",
        homepage="https://github.com/open-world-agents",
    )
    hf_dataset.info.update(info_to_update)

    # Convert to unified Dataset
    event_dataset = Dataset(
        arrow_table=hf_dataset.data,
        info=hf_dataset.info,
        split=hf_dataset.split,
        indices_table=hf_dataset._indices,
        fingerprint=hf_dataset._fingerprint,
        owa_config=DatasetConfig(
            stage=DatasetStage.EVENT,
            mcap_root_directory=mcap_root_directory,
        ),
    )

    return event_dataset
