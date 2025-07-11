#!/usr/bin/env python3
"""
Minimal training script for Multimodal LLM with FSLDataset using Hugging Face Accelerate.

This script demonstrates how to train a vision-language model on desktop interaction data
using the OWA FSLDataset format with proper multimodal handling.
"""

import argparse
from pathlib import Path

import torch
import wandb
from accelerate import Accelerator
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from owa.data.episode_tokenizer import EpisodeTokenizer
from owa.data.fsl_dataset import FSLDataset, FSLDatasetConfig


def collate_fn(batch):
    """Custom collate function for FSLDataset samples."""
    # Extract components from batch
    token_ids = [torch.tensor(sample["token_ids"], dtype=torch.long) for sample in batch]
    attention_masks = [torch.tensor(sample["attention_mask"], dtype=torch.long) for sample in batch]
    images = [sample["images"] for sample in batch]

    # Stack tensors (they should all be the same length due to padding)
    input_ids = torch.stack(token_ids)
    attention_mask = torch.stack(attention_masks)

    # Flatten images from all samples
    all_images = []
    for sample_images in images:
        all_images.extend(sample_images)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "images": all_images,
        "labels": input_ids.clone(),  # For causal language modeling
    }


def compute_loss(model, batch, processor):
    """Compute the training loss for a batch."""
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    images = batch["images"]
    labels = batch["labels"]

    # Process images if any
    pixel_values = None
    if images:
        # Convert PIL images to tensors
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels=labels,
    )

    return outputs.loss


def train_epoch(model, dataloader, optimizer, scheduler, accelerator, processor, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process)

    for step, batch in enumerate(progress_bar):
        with accelerator.accumulate(model):
            loss = compute_loss(model, batch, processor)

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            # Update progress bar
            avg_loss = total_loss / (step + 1)
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

            # Log to wandb
            if accelerator.is_local_main_process and step % 10 == 0:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/step": epoch * num_batches + step,
                    }
                )

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train MLLM with FSLDataset")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to event dataset")
    parser.add_argument(
        "--model-name", type=str, default="HuggingFaceTB/SmolVLM2-2.2B-Base", help="Model name or path"
    )
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max-sequence-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--logging-steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--wandb-project", type=str, default="owa-mllm-training", help="Wandb project name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")

    args = parser.parse_args()

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb" if not args.no_wandb else None,
        project_dir=args.output_dir,
    )

    # Initialize wandb
    if accelerator.is_local_main_process and not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
        )

    # Load tokenizer and processor
    accelerator.print(f"Loading tokenizer and processor from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    processor = AutoProcessor.from_pretrained(args.model_name)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and prepare dataset
    accelerator.print(f"Loading dataset from {args.dataset_path}")
    event_dataset = load_from_disk(args.dataset_path)

    # Initialize episode tokenizer
    episode_tokenizer = EpisodeTokenizer(image_token="<image>")
    episode_tokenizer.prepare_model(tokenizer=tokenizer)

    # Tokenize event dataset
    accelerator.print("Tokenizing event dataset...")
    for split, dataset in event_dataset.items():
        event_dataset[split] = episode_tokenizer.tokenize_event_dataset(dataset)

    # Create FSL dataset
    fsl_config = FSLDatasetConfig(
        pad_token_id=tokenizer.pad_token_id,
        max_sequence_length=args.max_sequence_length,
        load_images=True,
    )

    train_dataset = FSLDataset(event_dataset["train"], config=fsl_config)
    train_dataset.prepare()

    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with images
    )

    # Load model
    accelerator.print(f"Loading model from {args.model_name}")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if accelerator.device.type == "cuda" else torch.float32,
    )

    # Resize token embeddings for new tokens
    model.resize_token_embeddings(len(tokenizer))

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    num_training_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)

    # Training loop
    accelerator.print("Starting training...")
    for epoch in range(args.num_epochs):
        avg_loss = train_epoch(model, train_dataloader, optimizer, scheduler, accelerator, processor, epoch)

        accelerator.print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

        # Save checkpoint
        if accelerator.is_local_main_process:
            output_dir = Path(args.output_dir) / f"checkpoint-epoch-{epoch}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save model and tokenizer
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            accelerator.print(f"Checkpoint saved to {output_dir}")

    # Final save
    if accelerator.is_local_main_process:
        final_output_dir = Path(args.output_dir) / "final"
        final_output_dir.mkdir(parents=True, exist_ok=True)

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)

        accelerator.print(f"Final model saved to {final_output_dir}")

        if not args.no_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
