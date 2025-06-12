from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import typer
from accelerate.utils import set_seed
from datasets import DatasetDict

from dataset.build_dataset import (
    create_dataset,
    generate_conversation,
    generate_sampling_weight,
)

set_seed(23)
app = typer.Typer()


@app.command()
def main(
    train_path: Annotated[
        Path, typer.Option(help="Path to the train dataset directory", exists=True)
    ],
    output_path: Annotated[
        Path,
        typer.Option(help="Path to save the dataset, with jsonl ext.", exists=False),
    ],
    test_path: Annotated[
        Optional[Path], typer.Option(help="Path to the test dataset directory")
    ] = None,
):
    train_files = list(Path(train_path).rglob("*.mcap"))

    if test_path is None:
        # Use 90% of train data for training and 10% for testing
        generator = np.random.default_rng(23)
        indices = generator.permutation(len(train_files))
        split_idx = int(len(train_files) * 0.9)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        test_files = [train_files[i] for i in test_indices]
        train_files = [train_files[i] for i in train_indices]
        print(
            f"No test_path provided. Using {len(train_files)} files for training and {len(test_files)} files for testing."
        )
    else:
        test_files = list(Path(test_path).rglob("*.mcap"))

    dataset_dict: DatasetDict = create_dataset(train_files, test_files)

    for idx, data in enumerate(dataset_dict["train"].take(3)):
        print(data)  # Process the data as needed

    dataset_dict = generate_conversation(dataset_dict)
    # TODO: implement this
    # dataset_dict = generate_sampling_weight(dataset_dict)

    dataset_dict.save_to_disk(output_path)
    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    app()
