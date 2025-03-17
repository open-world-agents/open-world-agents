# This is example of SmolVLM training script
# Ref:
# https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm_smol_vlm.py
# https://github.com/huggingface/smollm/blob/main/vision/finetuning/Smol_VLM_FT.ipynb
# https://github.com/huggingface/smollm/blob/main/vision/finetuning/SmolVLM2_Video_FT.ipynb

"""
Reformated input(sample) of the dataset which tokenized by DiscreteTokenizer should be like this:
[
    {
        "state_keyboard": [keyboard, keyboard, ...],
        "state_mouse": [(timestamp, mouse), (timestamp, mouse), ...],
        "state_screen": [screen:PIL.Image, screen:PIL.Image, ...],
        "action_keyboard": [(timestamp, keyboard), (timestamp, keyboard), ...],
        "action_mouse": [(timestamp, mouse), (timestamp, mouse), ...],
    },
    ...
]

Note that keyboard contains the 1) virtual key code(Maximum 256 space size refer to window os) + 2)state of the key (pressed or not)
Also mouse contains the 1) x, y position + 2) event type (click, move, scroll) + 3) button (left, right, middle)
In this exmaple, we will not use state_mouse and action_mouse!



Verifying the bottleneck of training.
- Ignoring dataset bottleneck, assume 1it/s for batch size 32
- dataset is prepared 2 s/sample. So, required dataloader_num_workers to match model = 32 * 2 = 64
"""

import argparse
import functools
import os

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import Subset
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import SFTConfig, SFTTrainer

from owa_game_agent.data.datasets.smolvlm2 import SmolVLM2Dataset, collate_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SmolVLM model on OWA game queries")
    parser.add_argument("--query_path", type=str, required=True, help="Path to JSONL file containing queries")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save model checkpoints")
    parser.add_argument("--model_id", type=str, default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct", help="Model ID")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--save_steps", type=int, default=2000, help="Save checkpoint every X steps")
    args = parser.parse_args()

    accelerator = Accelerator()

    accelerator.print("Starting training process")

    ##########
    # Model, Tokenizer & Processor
    ##########
    accelerator.print(f"Loading model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    )

    ################
    # Dataset
    ################
    accelerator.print(f"Loading dataset from: {args.query_path}")
    dataset = SmolVLM2Dataset(args.query_path, processor=processor)

    # Split dataset into train and validation
    train_indices = np.random.choice(len(dataset), int(0.8 * len(dataset)), replace=False)
    eval_indices = np.setdiff1d(np.arange(len(dataset)), train_indices)

    # Define train and eval datasets
    train_dataset = Subset(dataset, train_indices)
    eval_dataset = Subset(dataset, eval_indices)

    ################
    # Training arguments
    ################
    os.makedirs(args.output_dir, exist_ok=True)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        learning_rate=args.learning_rate,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        max_seq_length=6144,
        report_to="wandb",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="",  # dummy field for collator
        dataset_kwargs={"skip_prepare_dataset": True},  # important for collator
        dataloader_num_workers=32,  # important for throughput
        run_name=os.path.basename(args.output_dir),
        remove_unused_columns=False,  # required to keep various state/action keys in the dataset
        # Evaluation settings
        do_eval=True,
        eval_strategy="epoch",
        eval_steps=1,
    )

    ################
    # Training
    ################
    accelerator.print("Initializing trainer")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=functools.partial(collate_fn, processor=processor),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor.tokenizer,
    )

    accelerator.print("Starting training")
    trainer.train()

    # Save and push to hub
    accelerator.print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    accelerator.print("Training completed")
