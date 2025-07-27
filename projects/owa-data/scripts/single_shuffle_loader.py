import numpy as np
from datasets import load_from_disk
from loguru import logger
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer

from owa.data.datasets import FSLDataset
from owa.data.episode_tokenizer import EpisodeTokenizer

# This line is to enable throughput logging from FSLDataset
logger.enable("owa.data.datasets.fsl_dataset")

# Load event dataset
event_dataset = load_from_disk("/mnt/raid12/datasets/owa/data/super-hexagon-event")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
image_processor = AutoImageProcessor.from_pretrained(
    "HuggingFaceTB/SmolVLM2-256M-Video-Instruct", do_image_splitting=False, use_fast=True
)

event_tokenizer = EpisodeTokenizer(image_token="<image>")
event_tokenizer.prepare_model(tokenizer=tokenizer)

for split, dataset in event_dataset.items():
    tokenized = event_tokenizer.tokenize_event_dataset(dataset)
    event_dataset[split] = tokenized


dataset = FSLDataset(
    event_dataset["train"],
    image_processor=image_processor,
    pad_token_id=tokenizer.pad_token_id,
    max_sequence_length=1024,
)
dataset.prepare()

for sample in dataset.take(1):
    print(f"{sample=}")

# take random shuffle
shuffled_index = np.random.permutation(len(dataset))
for i in tqdm(shuffled_index):  # expected: 2.1 it/s
    ...
