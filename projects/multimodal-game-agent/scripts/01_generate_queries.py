"""
This script's I/O

Input: list[mcap + mkv]
Output: list[query]
"""

from pathlib import Path

import numpy as np
import pandas as pd
import typer

from mcap_owa.highlevel import OWAMcapReader
from owa_game_agent.constants import RECORD_PAUSE_KEY, RECORD_START_STOP_KEY
from owa_game_agent.data import OWAMcapQuery, TimeUnits

app = typer.Typer()

PAST_RANGE_S = 0.25
FUTURE_RANGE_S = 0.25
FRAMERATE = 20  # 0.25 * 20 = 5, 5 image input


def sample_interval():
    return np.random.rand() * (PAST_RANGE_S + FUTURE_RANGE_S)


def extract_query(mcap_file: Path) -> list[OWAMcapQuery]:
    key_events = []
    with OWAMcapReader(mcap_file) as reader:
        for topic, timestamp, msg in reader.iter_decoded_messages(topics=["keyboard"]):
            if msg.event_type == "release" and msg.vk == RECORD_START_STOP_KEY:
                key_events.append(timestamp)
            elif msg.vk == RECORD_PAUSE_KEY:
                raise NotImplementedError("Pause key is not implemented")

        key_events = [reader.start_time, reader.end_time]  # TODO: remove this line

    assert len(key_events) % 2 == 0, "Number of start/stop key events should be even"
    intervals = np.array(key_events).reshape(-1, 2)

    queries = []
    for start, end in intervals:
        anchor = start + TimeUnits.SECOND * PAST_RANGE_S
        while anchor < end:
            query = OWAMcapQuery(
                file_path=mcap_file,
                anchor_timestamp_ns=anchor,
                past_range_ns=TimeUnits.SECOND * PAST_RANGE_S,
                future_range_ns=TimeUnits.SECOND * FUTURE_RANGE_S,
                screen_framerate=FRAMERATE,
            )
            try:
                query.to_sample()
                queries.append(query)
            except ValueError:
                pass
            anchor += TimeUnits.SECOND * sample_interval()

    return queries


@app.command()
def main(dataset_path: Path, query_path: Path):
    queries: dict[str, list[OWAMcapQuery]] = {}
    for mcap_file in dataset_path.glob("*.mcap"):
        queries[mcap_file] = extract_query(mcap_file)

    # print statistics for duration
    durations = []
    for mcap_file, qs in queries.items():
        with OWAMcapReader(mcap_file) as reader:
            durations.append(reader.duration)

    durations = pd.Series(durations) / TimeUnits.SECOND
    print("Duration statistics")
    print(durations.describe())

    # print statistics for number of queries
    num_queries = pd.Series([len(qs) for qs in queries.values()])
    print("Number of queries statistics")
    print(num_queries.describe())

    # save queries into query_path (jsonl file)
    all_queries: list[OWAMcapQuery] = []
    for qs in queries.values():
        all_queries.extend(qs)

    with open(query_path, "w") as f:
        for query in all_queries:
            f.write(query.model_dump_json() + "\n")


if __name__ == "__main__":
    app()
