#!/usr/bin/env python3
# multi_gpu_loader.py

import torch
from accelerate import Accelerator
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from owa.data.episode_tokenizer import EpisodeTokenizer
from owa.data.fsl_dataset import FSLDataset


def create_pretraining_collator(processor):
    def collate_fn(examples):
        # Extract data from FSLDataset format
        input_ids_list = []
        attention_mask_list = []
        images_list = []

        for example in examples:
            input_ids_list.append(example["token_ids"])
            attention_mask_list.append(example["attention_mask"])
            images_list.extend(example["images"])  # Flatten images from all examples

        # Convert to tensors
        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)

        # For pretraining, labels are the same as input_ids (next token prediction)
        # We shift the labels inside the model, so we don't need to do it here
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # Handle images - convert PIL images to tensors if needed. NOTE: smolvlm processor panic when image list is empty.
        if images_list:
            image_inputs = processor.image_processor(images_list, return_tensors="pt")
            pixel_values = image_inputs.get("pixel_values")
        else:
            pixel_values = None

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if pixel_values is not None:
            batch["pixel_values"] = pixel_values

        return batch

    return collate_fn


def main():
    # 1) Initialize Accelerator
    accelerator = Accelerator()

    # 2) Load & tokenize your event dataset
    print("▶ Loading raw dataset…")
    event_ds = load_from_disk("/mnt/raid12/datasets/owa/data/super-hexagon-event")

    print("▶ Loading tokenizer…")
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Base", do_image_splitting=False)

    print("▶ Preparing EpisodeTokenizer…")
    ep_tok = EpisodeTokenizer(image_token="<image>")
    ep_tok.prepare_model(tokenizer=processor.tokenizer)

    print("▶ Tokenizing splits…")
    tokenized = {}
    for split, ds in event_ds.items():
        tokenized[split] = ep_tok.tokenize_event_dataset(ds)

    # 3) Wrap into your FSLDataset (only train shown here)
    train_ds = FSLDataset(tokenized["train"], pad_token_id=processor.tokenizer.pad_token_id, max_sequence_length=1024)
    train_ds.prepare()

    # 4) Create a DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        collate_fn=create_pretraining_collator(processor),
    )

    # 5) (Optional) A dummy model so you can do a full prepare()
    model = torch.nn.Linear(1024, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 6) Let Accelerator wrap model, optimizer, and dataloader
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # 7) Simple loop to verify each GPU/process sees its shard
    for epoch in range(2):
        for step, batch in enumerate(train_loader):
            # batch["input_ids"] is on the correct device
            # (B, seq_len) → just do a dummy forward
            loss = model(batch["input_ids"].float()).mean()
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            if step % 20 == 0:
                accelerator.print(f"[Epoch {epoch} · step {step:4d}] loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
