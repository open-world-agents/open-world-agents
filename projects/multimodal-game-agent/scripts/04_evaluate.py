import argparse

import numpy as np
import torch
from torch.utils.data import Subset
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.smolvlm import SmolVLMProcessor

from owa_game_agent.data.datasets.smolvlm2 import SmolVLM2Dataset


def evaluate(model, processor: SmolVLMProcessor, dataset: SmolVLM2Dataset):
    # Split dataset into train and validation
    rng = np.random.default_rng(seed=23)  # Needed to reproduce the same split per each run / per each device
    train_indices = rng.choice(len(dataset), int(0.8 * len(dataset)), replace=False)
    eval_indices = np.setdiff1d(np.arange(len(dataset)), train_indices)

    # Define train and eval datasets
    train_dataset = Subset(dataset, train_indices)
    eval_dataset = Subset(dataset, eval_indices)

    examples = [eval_dataset[i] for i in range(8)]
    evaluate_step(model, processor, examples)


def evaluate_step(model, processor: SmolVLMProcessor, examples):
    texts = []
    labels = []
    images = []

    for ex in examples:
        assistant_prompt = ex["messages"].pop(-1)
        # BUG: Surprisingly, SmolVLM processor does NOT append "Assistant: " as generation prompt, but append "Assistant:"!
        # SEVERE bug because single space can be critical for the model to generate the correct response.
        texts.append(processor.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=True) + " ")
        labels.append(assistant_prompt["content"][0]["text"])
        images.append(ex["images"])

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True).to(model.device, dtype=model.dtype)

    outputs = model.generate(**batch, do_sample=False, max_new_tokens=64)

    for i, (text, label, output) in enumerate(zip(texts, labels, outputs)):
        generated: str = processor.decode(output)
        generated = generated[generated.find("Assistant: ") + len("Assistant: ") :]

        print(f"Example {i}")
        print(f"Text: {text}")
        print(f"Label: {label}")
        print(f"Output: {generated}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SmolVLM model on OWA game queries")
    parser.add_argument("--query_path", type=str, required=True, help="Path to JSONL file containing queries")
    parser.add_argument("--model_id", type=str, default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct", help="Model ID")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    ).to("cuda")

    dataset = SmolVLM2Dataset(args.query_path)

    evaluate(model=model, processor=processor, dataset=dataset)
