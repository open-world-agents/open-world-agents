"""
This script's I/O

Input: list[query]
Output: N/A
"""

from pathlib import Path

import typer
from PIL import Image

from owa_game_agent.data import OWAMcapQuery, OWATrainingSample

app = typer.Typer()


def tokenize_sample(sample: OWATrainingSample) -> OWATrainingSample:
    # TODO: implement this function
    ...


class SmolVLMInput:
    images: list[Image.Image]
    messages: dict


def sample_to_smolvlm_input(sample: OWATrainingSample) -> SmolVLMInput:
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
