from pathlib import Path
from typing import Annotated

import typer
from dataset.build_dataset import create_dataset

app = typer.Typer()


@app.command()
def main(
    dataset_path: Annotated[Path, typer.Option(help="Path to the dataset directory", exists=True)],
    output_path: Annotated[Path, typer.Option(help="Path to save the dataset", exists=False)],
):
    dataset = create_dataset(dataset_path)
    for idx, data in enumerate(dataset):
        print(data)  # Process the data as needed
        if idx > 3:
            break


if __name__ == "__main__":
    app()
