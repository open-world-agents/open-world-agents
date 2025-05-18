from pathlib import Path
from typing import Annotated

import typer
from accelerate.utils import set_seed
from dataset.build_dataset import create_dataset, generate_conversation
from datasets import Dataset

set_seed(23)
app = typer.Typer()


@app.command()
def main(
    dataset_path: Annotated[Path, typer.Option(help="Path to the dataset directory", exists=True)],
    output_path: Annotated[Path, typer.Option(help="Path to save the dataset, with jsonl ext.", exists=False)],
):
    dataset: Dataset = create_dataset(dataset_path)
    for idx, data in enumerate(dataset.take(3)):
        print(data)  # Process the data as needed

    dataset = generate_conversation(dataset)

    dataset.save_to_disk(output_path)


if __name__ == "__main__":
    app()
