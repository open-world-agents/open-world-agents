from pathlib import Path
from typing import Annotated

import typer
from accelerate.utils import set_seed
from datasets import DatasetDict

from dataset.build_dataset import create_dataset, generate_conversation, generate_sampling_weight

set_seed(23)
app = typer.Typer()


@app.command()
def main(
    train_path: Annotated[Path, typer.Option(help="Path to the train dataset directory", exists=True)],
    test_path: Annotated[Path, typer.Option(help="Path to the test dataset directory", exists=True)],
    output_path: Annotated[Path, typer.Option(help="Path to save the dataset, with jsonl ext.", exists=False)],
):
    dataset_dict: DatasetDict = create_dataset(train_path, test_path)

    for idx, data in enumerate(dataset_dict["train"].take(3)):
        print(data)  # Process the data as needed

    dataset_dict = generate_conversation(dataset_dict)
    # TODO: implement this
    # dataset_dict = generate_sampling_weight(dataset_dict)

    dataset_dict.save_to_disk(output_path)
    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    app()
