import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, cast

from datasets import Dataset as HFDataset
from datasets import Features, Value
from tqdm import tqdm

from mcap_owa.highlevel import OWAMcapReader
from owa.data.datasets import Dataset, DatasetConfig, DatasetStage
from owa.data.interval.selector import InactivityFilter
from owa.data.processing.resampler import EventResamplerDict


@dataclass
class McapToEventConfig:
    """Configuration for MCAP to events conversion."""

    rate_settings: Dict[str, float]  # Mapping from topic to desired rate (Hz) for resampling
    keep_topics: Optional[List[str]] = None  # Optional list of topics to keep. If None, all topics are kept
    num_workers: int = 4  # Number of worker processes for parallel file processing
    interval_extractor: InactivityFilter = field(default_factory=InactivityFilter)


def _mcap_to_events(
    episode_path: str,
    config: McapToEventConfig,
    mcap_root_directory: Optional[str] = None,
) -> List[Dict]:
    """
    Process MCAP file with resampling.

    Args:
        episode_path: Path to the MCAP file to process
        config: Configuration object containing rate settings, topics, and other parameters
        mcap_root_directory: Optional root directory for storing relative paths

    Returns:
        List of event dictionaries containing processed events
    """
    events: List[Dict] = []
    valid_intervals = config.interval_extractor.extract_intervals(Path(episode_path))

    with OWAMcapReader(Path(episode_path)) as reader:
        for interval in valid_intervals:
            # Initialize resamplers for all topics. NOTE: resampler init must be here
            resamplers = EventResamplerDict(config.rate_settings)
            for mcap_msg in reader.iter_messages(
                start_time=interval.start, end_time=interval.end, topics=config.keep_topics
            ):
                # Process event through resampler
                resamplers.add_event(mcap_msg)
                resamplers.step(mcap_msg.timestamp)

                # Process all ready events
                for mcap_message_obj in resamplers.pop_events():
                    # Serialize McapMessage to bytes using model_dump_json
                    mcap_message_bytes = mcap_message_obj.model_dump_json().encode("utf-8")

                    # Store relative path if mcap_root_directory is provided
                    stored_episode_path = episode_path
                    if mcap_root_directory:
                        stored_episode_path = Path(episode_path).relative_to(mcap_root_directory).as_posix()

                    events.append(
                        {
                            "episode_path": stored_episode_path,
                            "topic": mcap_message_obj.topic,
                            "timestamp_ns": mcap_message_obj.timestamp,
                            "message_type": mcap_message_obj.message_type,
                            "mcap_message": mcap_message_bytes,  # Store serialized bytes
                        }
                    )

    return events


def _yield_events(
    episode_paths: List[str],
    config: McapToEventConfig,
    mcap_root_directory: Optional[str] = None,
    on_error: Optional[Callable[[str, BaseException], None]] = None,
) -> Generator[Dict, None, None]:
    """
    Generator function that yields event examples by processing each raw events file
    in parallel using multiple processes. Events within same mcap file is grouped together.

    Args:
        episode_paths: List of MCAP file paths (strings).
        config: Configuration object containing rate settings, topics, and other parameters.
        mcap_root_directory: Optional root directory for storing relative paths.
        on_error: Optional callback function to handle errors.

    Yields:
        Individual event dictionaries suitable for Hugging Face Dataset.
    """
    total_files = len(episode_paths)
    with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        future_to_path = {
            executor.submit(_mcap_to_events, fp, config, mcap_root_directory): fp for fp in episode_paths
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


def build_event_dataset(
    episode_paths: List[Path], *, config: McapToEventConfig, mcap_root_directory: Optional[str] = None
) -> Dataset:
    """
    Create a Hugging Face event dataset from the given MCAP file paths by streaming
    examples from a generator.

    Args:
        episode_paths: List of pathlib.Path objects pointing to MCAP files.
        config: Configuration object containing rate settings, topics, and other parameters.
        mcap_root_directory: Optional root directory for storing relative paths.

    Returns:
        A Hugging Face Dataset containing the combined events.
    """

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
        _yield_events,
        gen_kwargs={
            "episode_paths": episode_path_strs,
            "config": config,
            "mcap_root_directory": mcap_root_directory,
        },
        features=features,
    )
    hf_dataset = cast(HFDataset, hf_dataset)

    # Convert to unified Dataset using from_hf_dataset method
    owa_config = DatasetConfig(
        stage=DatasetStage.EVENT,
        mcap_root_directory=mcap_root_directory,
    )
    event_dataset = Dataset.from_hf_dataset(hf_dataset, owa_config=owa_config)

    return event_dataset
