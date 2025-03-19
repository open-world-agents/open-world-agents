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
    evaluate_step(model=model, processor=processor, examples=examples)


def logits_processor(input_ids: torch.LongTensor, scores: torch.FloatTensor):
    # print the difference between
    # 1. topk scores/index from 49279 to end
    # 2. topk scores/index from 0 to 49279

    # Number of top items to compare
    k = 5

    # Get topk scores and indices for the first segment (0 to 49279)
    first_segment = scores[:, :49279]
    first_topk_scores, first_topk_indices = torch.topk(first_segment, k)

    # Get topk scores and indices for the second segment (49279 to end)
    second_segment = scores[:, 49279:]
    second_topk_scores, second_topk_indices = torch.topk(second_segment, k)
    # Adjust indices to account for the offset
    second_topk_indices = second_topk_indices + 49279

    # Print comparison for each example in the batch
    for batch_idx in range(scores.shape[0]):
        print(f"\nBatch item {batch_idx} comparison:")
        print(f"Top {k} tokens from first segment (0-49279):")
        for i in range(k):
            token_id = first_topk_indices[batch_idx, i].item()
            token_score = first_topk_scores[batch_idx, i].item()
            print(f"  Token ID: {token_id}, Score: {token_score:.4f}")

        print(f"Top {k} tokens from second segment (49279+):")
        for i in range(k):
            token_id = second_topk_indices[batch_idx, i].item()
            token_score = second_topk_scores[batch_idx, i].item()
            print(f"  Token ID: {token_id}, Score: {token_score:.4f}")

    # Zero out the probability for token ID 49279 as in the original code
    scores[:, 49279] = 0
    return scores


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

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    # print(f"attention_mask: {attention_mask.tolist()}")

    outputs = model.generate(
        **batch,
        logits_processor=[logits_processor],
        do_sample=False,
        max_new_tokens=64,
        eos_token_id=processor.tokenizer.encode("<end_of_utterance>")[0],
    )

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

    processor = AutoProcessor.from_pretrained(args.model_id, padding_side="left")
    # https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/discussions/16
    processor.tokenizer.padding_side = "left"

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    ).to("cuda")

    dataset = SmolVLM2Dataset(args.query_path)

    evaluate(model=model, processor=processor, dataset=dataset)
