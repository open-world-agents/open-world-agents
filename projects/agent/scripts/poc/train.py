#!/usr/bin/env python3
"""
Minimal training script for Multimodal LLM with FSLDataset using Lightning Fabric.

This script demonstrates how to train a vision-language model on desktop interaction data
using the FSLDataset for efficient sequence packing and Lightning Fabric for distributed training.
"""

import argparse
import os

import torch
from datasets import load_from_disk
from lightning.fabric import Fabric
from lightning.fabric.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

from owa.data.episode_tokenizer import EpisodeTokenizer
from owa.data.fsl_dataset import FSLDataset, FSLDatasetConfig


def collate_fn(batch):
    """Custom collate function for FSLDataset samples."""
    # Extract data from batch
    token_ids = [torch.tensor(sample["token_ids"], dtype=torch.long) for sample in batch]
    attention_masks = [torch.tensor(sample["attention_mask"], dtype=torch.long) for sample in batch]
    images = [sample["images"] for sample in batch]

    # Stack token_ids and attention_masks
    token_ids = torch.stack(token_ids)
    attention_masks = torch.stack(attention_masks)

    # Flatten images list (each sample can have multiple images)
    all_images = []
    image_indices = []  # Track which sample each image belongs to
    for i, sample_images in enumerate(images):
        for img in sample_images:
            all_images.append(img)
            image_indices.append(i)

    return {
        "input_ids": token_ids,
        "attention_mask": attention_masks,
        "images": all_images,
        "image_indices": torch.tensor(image_indices, dtype=torch.long)
        if all_images
        else torch.empty(0, dtype=torch.long),
    }


def compute_loss(model, batch, processor):
    """Compute training loss for vision-language model."""
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    images = batch["images"]

    # Process images if any
    if images:
        # Convert PIL images to tensors using processor
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(input_ids.device)
    else:
        pixel_values = None

    # Create labels for causal language modeling (shift input_ids)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100  # Ignore padding tokens in loss

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels=labels,
    )

    return outputs.loss


def train_epoch(fabric, model, dataloader, optimizer, processor, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(progress_bar):
        # Zero gradients
        optimizer.zero_grad()

        # Compute loss
        loss = compute_loss(model, batch, processor)

        # Backward pass
        fabric.backward(loss)

        # Optimizer step
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        avg_loss = total_loss / num_batches

        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{avg_loss:.4f}"})

        # Log metrics
        if batch_idx % 10 == 0:
            fabric.log("train_loss", loss.item(), step=epoch * len(dataloader) + batch_idx)

    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train MLLM with FSLDataset")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to event dataset")
    parser.add_argument(
        "--model-name", type=str, default="HuggingFaceTB/SmolVLM2-2.2B-Base", help="Pretrained model name"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--max-sequence-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Training precision")

    args = parser.parse_args()

    # Initialize Fabric
    logger = TensorBoardLogger(root_dir="./logs")
    fabric = Fabric(
        accelerator="auto",
        devices=args.devices,
        precision=args.precision,
        loggers=logger,
    )
    fabric.launch()

    # Load dataset
    fabric.print(f"Loading dataset from {args.dataset_path}")
    event_dataset = load_from_disk(args.dataset_path)

    # Initialize tokenizer and processor
    fabric.print(f"Loading model and tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForVision2Seq.from_pretrained(args.model_name)

    # Setup episode tokenizer
    episode_tokenizer = EpisodeTokenizer(image_token="<image>")
    episode_tokenizer.prepare_model(tokenizer=tokenizer, model=model)

    # Tokenize event dataset
    fabric.print("Tokenizing event dataset...")
    for split, dataset in event_dataset.items():
        tokenized = episode_tokenizer.tokenize_event_dataset(dataset, map_kwargs={"num_proc": 4})
        event_dataset[split] = tokenized

    # Create FSLDataset
    fabric.print("Creating FSLDataset...")
    fsl_config = FSLDatasetConfig(
        max_sequence_length=args.max_sequence_length,
        pad_token_id=tokenizer.pad_token_id,
        load_images=True,
    )

    train_dataset = FSLDataset(event_dataset["train"], config=fsl_config)
    train_dataset.prepare()

    fabric.print(f"Dataset prepared with {len(train_dataset)} sequences")

    # Create DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Setup with Fabric
    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    # Training loop
    fabric.print("Starting training...")
    for epoch in range(args.epochs):
        avg_loss = train_epoch(fabric, model, train_dataloader, optimizer, processor, epoch)
        fabric.print(f"Epoch {epoch + 1}/{args.epochs} - Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        if fabric.global_rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            fabric.save(checkpoint_path, {"model": model, "optimizer": optimizer, "epoch": epoch})
            fabric.print(f"Checkpoint saved: {checkpoint_path}")

    fabric.print("Training completed!")


if __name__ == "__main__":
    main()
