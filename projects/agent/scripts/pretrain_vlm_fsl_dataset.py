# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pretraining script for Vision-Language Models using FSLDataset.

This script enables pretraining of multimodal models on desktop interaction data
using the FSLDataset format for efficient sequence packing and tokenization-aware training.

Example usage:
# Using config file (recommended):
accelerate launch --config_file=accelerate_configs/deepspeed_zero1.yaml \
    pretrain_vlm_fsl_dataset.py \
    --config pretrain_training_config.yaml

# Using command line arguments:
accelerate launch --config_file=accelerate_configs/deepspeed_zero1.yaml \
    pretrain_vlm_fsl_dataset.py \
    --dataset_path /path/to/event/dataset \
    --model_name_or_path HuggingFaceTB/SmolVLM2-256M-Video-Instruct \
    --output_dir pretrain-smol-vlm-fsl \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_sequence_length 1024 \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --learning_rate 3e-4 \
    --num_train_epochs 3 \
    --save_steps 1000 \
    --logging_steps 100
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_from_disk
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_quantization_config,
)

from owa.data.episode_tokenizer import EpisodeTokenizer
from owa.data.fsl_dataset import FSLDataset, FSLDatasetConfig


@dataclass
class PretrainScriptArguments(ScriptArguments):
    """
    Arguments for pretraining script using FSLDataset.
    """

    dataset_path: str = field(default="", metadata={"help": "Path to the event dataset directory"})
    max_sequence_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length for FSLDataset, note that SFTConfig's max_length/max_seq_length already exists"
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )


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
    # Initialize accelerator for distributed training
    accelerator = Accelerator()

    parser = TrlParser((PretrainScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Configure training arguments for pretraining
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    ################
    # Model, Tokenizer & Processor
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # Load processor and tokenizer
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    tokenizer = processor.tokenizer  # TODO: save this

    # Set tokenizer model_max_length if needed
    if hasattr(tokenizer, "model_max_length"):
        accelerator.print(
            f"Tokenizer model_max_length: {tokenizer.model_max_length}, using max_sequence_length: {script_args.max_sequence_length}"
        )

    # Load model
    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    # Check max_position_embeddings for the text model
    if hasattr(model, "model") and hasattr(model.model, "text_model") and hasattr(model.model.text_model, "config"):
        text_config = model.model.text_model.config
        if hasattr(text_config, "max_position_embeddings"):
            original_max_pos = text_config.max_position_embeddings
            if script_args.max_sequence_length > original_max_pos:
                accelerator.print(
                    f"Warning: Requested max_sequence_length ({script_args.max_sequence_length}) is larger than "
                    f"model's max_position_embeddings ({original_max_pos}). This may cause issues."
                )
            else:
                accelerator.print(
                    f"Model max_position_embeddings: {original_max_pos}, using max_sequence_length: {script_args.max_sequence_length}"
                )

    # Print trainable parameters
    trainable_params = model.num_parameters(only_trainable=True)
    total_params = model.num_parameters()
    accelerator.print(
        f"Model has {trainable_params:,} / {total_params:,} ({trainable_params / total_params:.2%}) trainable parameters"
    )

    ################
    # Dataset Preparation
    ################
    accelerator.print(f"Loading dataset from {script_args.dataset_path}...")
    event_dataset = load_from_disk(script_args.dataset_path)

    # Setup episode tokenizer
    episode_tokenizer = EpisodeTokenizer(image_token="<image>")
    episode_tokenizer.prepare_model(tokenizer=tokenizer, model=model)

    # Tokenize event dataset
    accelerator.print("Tokenizing event dataset...")
    # Handle both DatasetDict and single Dataset cases
    if hasattr(event_dataset, "keys") and callable(getattr(event_dataset, "keys")):
        # DatasetDict case
        tokenized_datasets = {}
        for split_name in ["train", "test", "validation"]:
            if split_name in event_dataset:
                from datasets import Dataset

                dataset = event_dataset[split_name]
                # Type cast to help with type checking
                if isinstance(dataset, Dataset):
                    tokenized = episode_tokenizer.tokenize_event_dataset(
                        dataset, map_kwargs={"num_proc": script_args.preprocessing_num_workers}
                    )
                    tokenized_datasets[split_name] = tokenized
        event_dataset = tokenized_datasets
    else:
        # Single Dataset case - assume it's the train split
        from datasets import Dataset

        if isinstance(event_dataset, Dataset):
            tokenized = episode_tokenizer.tokenize_event_dataset(
                event_dataset, map_kwargs={"num_proc": script_args.preprocessing_num_workers}
            )
            event_dataset = {"train": tokenized}
        else:
            raise ValueError(f"Unsupported dataset type: {type(event_dataset)}")

    # Create FSL datasets
    accelerator.print("Creating FSLDataset...")
    fsl_config = FSLDatasetConfig(
        max_sequence_length=script_args.max_sequence_length,
        pad_token_id=tokenizer.pad_token_id,
        load_images=True,
    )

    train_fsl_dataset = FSLDataset(event_dataset["train"], config=fsl_config)
    train_fsl_dataset.prepare()

    eval_fsl_dataset = None
    if "test" in event_dataset:
        eval_fsl_dataset = FSLDataset(event_dataset["test"], config=fsl_config)
        eval_fsl_dataset.prepare()

    ################
    # Data Collator
    ################
    data_collator = create_pretraining_collator(processor)

    ################
    # Trainer
    ################
    accelerator.print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_fsl_dataset,
        eval_dataset=eval_fsl_dataset,
        data_collator=data_collator,
        processing_class=processor,
    )

    ################
    # Training
    ################
    accelerator.print("Starting training...")
    trainer.train()

    # Save final model
    accelerator.print("Saving final model...")
    trainer.save_model(training_args.output_dir)

    # Push to hub if requested
    if training_args.push_to_hub:
        accelerator.print("Pushing to hub...")
        trainer.push_to_hub(dataset_name=script_args.dataset_path)
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)

    accelerator.print("Training completed!")


if __name__ == "__main__":
    main()
