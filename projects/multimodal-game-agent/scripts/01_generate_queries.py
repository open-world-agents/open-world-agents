"""
This script's I/O

Input: list[mcap] (dataset_path)
Output: list[query] (query_path)
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from typing_extensions import Annotated
from accelerate.utils import set_seed
from tqdm import tqdm

from mcap_owa.highlevel import OWAMcapReader
from owa.core.time import TimeUnits
from owa_game_agent.constants import RECORD_PAUSE_KEY, RECORD_START_STOP_KEY
from owa_game_agent.data import OWAMcapQuery

set_seed(23)

app = typer.Typer()

PAST_RANGE_S = 0.25
FUTURE_RANGE_S = 1
FRAMERATE = 20  # 0.25 * 20 = 5, 5 image input


def sample_interval():
    return np.random.rand() * (PAST_RANGE_S + FUTURE_RANGE_S)


def process_query(query: OWAMcapQuery) -> OWAMcapQuery:
    """Function to process one query in parallel."""
    try:
        query.to_sample()
        return query
    except ValueError:
        return None


def extract_query(mcap_file: Path) -> list[OWAMcapQuery]:
    key_events = []
    with OWAMcapReader(mcap_file) as reader:
        for topic, timestamp, msg in reader.iter_decoded_messages(topics=["keyboard"]):
            if msg.event_type == "release" and msg.vk == RECORD_START_STOP_KEY:
                key_events.append(timestamp)
            elif msg.vk == RECORD_PAUSE_KEY:
                raise NotImplementedError("Pause key is not implemented")

    assert len(key_events) % 2 == 0, "Number of start/stop key events should be even"
    intervals = np.array(key_events).reshape(-1, 2)

    # Filter only intervals longer than 60 seconds (SUPER HEXAGON)
    intervals = intervals[intervals[:, 1] - intervals[:, 0] > TimeUnits.SECOND * 60]

    pbar = tqdm(
        intervals,
        desc=f"Extracting queries from {mcap_file.name} with {len(intervals)} intervals",
    )

    query_list = []
    for start, end in intervals:
        anchor = start + TimeUnits.SECOND * PAST_RANGE_S
        while anchor < end:
            pbar.update()
            query = OWAMcapQuery(
                file_path=mcap_file,
                anchor_timestamp_ns=anchor,
                past_range_ns=TimeUnits.SECOND * PAST_RANGE_S,
                future_range_ns=TimeUnits.SECOND * FUTURE_RANGE_S,
                screen_framerate=FRAMERATE,
            )
            query_list.append(query)
            anchor += TimeUnits.SECOND * sample_interval()

    queries = []
    # Parallelizing query.to_sample()
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_query, query): query for query in query_list}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing queries"
        ):
            result = future.result()
            if result:
                queries.append(result)

    return queries


@app.command()
def main(
    dataset_path: Annotated[Path, typer.Option("--dataset_path")],
    query_path: Annotated[Path, typer.Option("--query_path")],
):
    queries: dict[Path, list[OWAMcapQuery]] = {}
    for mcap_file in dataset_path.glob("*.mcap"):
        queries[mcap_file] = extract_query(mcap_file)

    # Print statistics for duration
    durations = []
    for mcap_file, qs in queries.items():
        with OWAMcapReader(mcap_file) as reader:
            durations.append(reader.duration)

    durations = pd.Series(durations) / TimeUnits.SECOND
    print("Duration statistics")
    print(durations.describe())

    # Print statistics for number of queries
    num_queries = pd.Series([len(qs) for qs in queries.values()])
    print("Number of queries statistics")
    print(num_queries.describe())

    # Save queries into query_path (jsonl file)
    all_queries: list[OWAMcapQuery] = []
    for qs in queries.values():
        all_queries.extend(qs)

    with open(query_path, "w") as f:
        for query in all_queries:
            f.write(query.model_dump_json() + "\n")


if __name__ == "__main__":
    app()
