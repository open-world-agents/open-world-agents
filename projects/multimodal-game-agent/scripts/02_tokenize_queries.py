"""
This script's I/O

Input: list[query]
Output: N/A
"""

from pathlib import Path

import numpy as np
import orjson
import pandas as pd
import typer

from mcap_owa.highlevel import OWAMcapReader
from owa_game_agent.constants import RECORD_PAUSE_KEY, RECORD_START_STOP_KEY
from owa_game_agent.data import OWAMcapQuery, OWATrainingSample, TimeUnits

app = typer.Typer()


def tokenize_sample(sample: OWATrainingSample) -> OWATrainingSample:
    # TODO: implement this function
    ...


def sample_to_smolvlm_input(sample: OWATrainingSample) -> dict:
    # TODO: implement this function
    ...


@app.command()
def main(query_path: Path):
    # for each query, extract the sample.
    # query_path is jsonl file
    with open(query_path, "r") as f:
        queries = [OWAMcapQuery.model_validate_json(line) for line in f]

    # TODO: implement following
    query = queries[0]
    sample = query.to_sample()
    print(sample)

    sample = tokenize_sample(sample)
    sample = sample_to_smolvlm_input(sample)


if __name__ == "__main__":
    app()
