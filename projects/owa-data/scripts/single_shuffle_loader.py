import numpy as np
from loguru import logger
from tqdm import tqdm
from transformers import AutoImageProcessor

from owa.data.datasets import load_from_disk

# This line is to enable throughput logging from FSLDataset
logger.enable("owa.data.datasets.fsl_dataset")

# Load FSL dataset (pre-computed)
dataset = load_from_disk("/raid/datasets/owa/data/csgo-fsl")
image_processor = AutoImageProcessor.from_pretrained(
    "HuggingFaceTB/SmolVLM2-256M-Video-Instruct", do_image_splitting=False, use_fast=True
)

# Apply FSL transform for on-the-fly processing
train_dataset = dataset["train"]
train_dataset.auto_set_transform(stage="fsl", load_images=True, image_processor=image_processor)

for sample in train_dataset.take(1):
    print(f"{sample=}")

# take random shuffle
shuffled_index = np.random.permutation(len(train_dataset))
for i in tqdm(shuffled_index):  # expected: 2.1 it/s
    sample = train_dataset[int(i)]
