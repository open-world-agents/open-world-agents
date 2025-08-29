import argparse

import numpy as np
from loguru import logger
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoProcessor

from owa.data.collator import ModelType, detect_model_type
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

    model_type = detect_model_type(args.model)
    print(f"Detected model type: {model_type}")

    # Configure processor based on model type
    if model_type == ModelType.INTERNVL:
        # InternVL3 configuration: disable multi-crop for efficiency
        processor = AutoProcessor.from_pretrained(args.model)
        processor.image_processor = AutoImageProcessor.from_pretrained(
            args.model,
            use_fast=True,
            crop_to_patches=False,  # InternVL3-specific: disable multi-crop
        )
        print("Configured InternVL3 processor with multi-crop disabled")
    else:
        # SmolVLM2 and other models configuration
        processor = AutoProcessor.from_pretrained(args.model, do_image_splitting=False)
        processor.image_processor = AutoImageProcessor.from_pretrained(
            args.model,
            use_fast=True,
            do_image_splitting=False,  # SmolVLM2-specific: disable image splitting
        )
        print("Configured SmolVLM2/default processor")

    # Load and process datasets
    train_datasets = []
    for dataset_path in args.datasets:
        logger.info(f"Loading dataset from: {dataset_path}")
        dataset = load_from_disk(dataset_path)
        train_dataset = dataset["train"]
        train_dataset.auto_set_transform(stage="fsl", load_images=True, image_processor=processor.image_processor)
        train_datasets.append(train_dataset)

    # Concatenate all datasets
    train_dataset = ConcatDataset(train_datasets)

    # Print sample for verification
    for sample in train_dataset:
        print(f"{sample=}")
        break

    # Take random shuffle
    shuffled_index = np.random.permutation(len(train_dataset))
    original_index = np.arange(len(train_dataset))  # noqa: F841
    for i in tqdm(shuffled_index):  # expected: 2.1 it/s
        sample = train_dataset[int(i)]


if __name__ == "__main__":
    main()
