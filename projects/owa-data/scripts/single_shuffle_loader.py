import argparse

import numpy as np
from loguru import logger
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from transformers import AutoImageProcessor

from owa.data.datasets import load_from_disk

# This line is to enable throughput logging from FSLTransform
logger.enable("owa.data.datasets.transforms")


def main():
    parser = argparse.ArgumentParser(description="Load and shuffle FSL datasets")
    parser.add_argument(
        "datasets",
        nargs="+",
        help="List of dataset paths to load (e.g., /path/to/dataset1 /path/to/dataset2)",
    )
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        help="Model name for image processor (default: HuggingFaceTB/SmolVLM2-256M-Video-Instruct)",
    )

    args = parser.parse_args()

    # Load image processor
    image_processor = AutoImageProcessor.from_pretrained(args.model, do_image_splitting=False, use_fast=True)

    # Load and process datasets
    train_datasets = []
    for dataset_path in args.datasets:
        logger.info(f"Loading dataset from: {dataset_path}")
        dataset = load_from_disk(dataset_path)
        train_dataset = dataset["train"]
        train_dataset.auto_set_transform(stage="fsl", load_images=True, image_processor=image_processor)
        train_datasets.append(train_dataset)

    # Concatenate all datasets
    train_dataset = ConcatDataset(train_datasets)

    # Print sample for verification
    for sample in train_dataset:
        print(f"{sample=}")
        break

    # Take random shuffle
    shuffled_index = np.random.permutation(len(train_dataset))
    for i in tqdm(shuffled_index):  # expected: 2.1 it/s
        sample = train_dataset[int(i)]


if __name__ == "__main__":
    main()
