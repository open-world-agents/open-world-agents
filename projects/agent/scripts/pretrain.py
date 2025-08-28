# Copied from https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm_smol_vlm.py

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
using pre-computed FSLDataset format for efficient training. Supports multiple datasets.

Example usage:
# Using config file (recommended):
accelerate launch --config_file=accelerate_configs/deepspeed_zero1.yaml \
    pretrain.py \
    --config pretrain_config.yaml

# Using command line arguments with multiple datasets:
accelerate launch --config_file=accelerate_configs/deepspeed_zero1.yaml \
    pretrain.py \
    --dataset_paths /path/to/fsl/dataset1 /path/to/fsl/dataset2 \
    --model_name_or_path HuggingFaceTB/SmolVLM2-256M-Video-Instruct \
    --output_dir pretrain-smol-vlm-fsl \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --learning_rate 3e-4 \
    --num_train_epochs 3 \
    --save_steps 1000 \
    --logging_steps 100
"""

from dataclasses import dataclass, field
from typing import List, cast

import torch
from accelerate import Accelerator
from loguru import logger  # noqa: F401
from torch.utils.data import ConcatDataset
from transformers import AutoImageProcessor, AutoModelForImageTextToText, AutoProcessor
from trl import (
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_quantization_config,
)

from owa.agent.training import OWASFTConfig as SFTConfig
from owa.agent.training import OWASFTTrainer as SFTTrainer
from owa.data.collator import ModelType, detect_model_type, get_collate_fn
from owa.data.datasets import Dataset, load_from_disk
from owa.data.episode_tokenizer import EpisodeTokenizer

# This line is to enable throughput logging from FSLTransform
# logger.enable("owa.data.datasets.transforms")


@dataclass
class PretrainScriptArguments(ScriptArguments):
    """
    Arguments for pretraining script using FSLDataset.
    """

    dataset_paths: List[str] = field(
        default_factory=list, metadata={"help": "List of paths to event dataset directories"}
    )
    max_sequence_length: int = field(default=1024, metadata={"help": "Maximum sequence length for FSLDataset"})


def limit_dataset(dataset, max_count=64):
    """Limit dataset count and convert back to OWA Dataset class."""
    limited_count = min(len(dataset), max_count)
    owa_config = dataset.owa_config
    return Dataset.from_hf_dataset(dataset.select(range(limited_count)), owa_config=owa_config)


def main():
    # Initialize accelerator for distributed training
    accelerator = Accelerator()

    parser = TrlParser((PretrainScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Configure training arguments for pretraining
    training_args = cast(SFTConfig, training_args)
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

    # Detect model type for appropriate configuration
    model_type = detect_model_type(model_args.model_name_or_path)
    accelerator.print(f"Detected model type: {model_type}")

    # Load processor and tokenizer with model-specific configuration
    if model_type == ModelType.INTERNVL:
        # InternVL3 configuration: disable multi-crop for efficiency
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, crop_to_patches=False
        )
        assert processor.image_processor.crop_to_patches is False, "Failed to disable multi-crop"
        assert processor.image_processor.__class__.__name__ == "GotOcr2ImageProcessorFast", (
            f"Expected GotOcr2ImageProcessorFast, got {processor.image_processor.__class__}"
        )
        accelerator.print("Configured InternVL3 processor with multi-crop disabled")
    else:
        # SmolVLM2 and other models configuration
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, do_image_splitting=False
        )
        processor.image_processor = AutoImageProcessor.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            do_image_splitting=False,
            use_fast=True,
        )
        assert processor.image_processor.do_image_splitting is False, "Failed to disable image splitting"
        assert processor.image_processor.__class__.__name__ == "SmolVLMImageProcessorFast", (
            f"Expected SmolVLMImageProcessorFast, got {processor.image_processor.__class__}"
        )
        accelerator.print("Configured SmolVLM2/default processor")

    tokenizer = processor.tokenizer

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

    # Expand vocab. NOTE: take care of here when you switch model / event encoder
    episode_tokenizer = EpisodeTokenizer.from_transformers_model(model_args.model_name_or_path)
    episode_tokenizer.prepare_model(tokenizer=tokenizer, model=model)

    # Freeze vision encoder TODO: make this configurable
    # if hasattr(model, "model") and hasattr(model.model, "vision_model"):
    #     for param in model.model.vision_model.parameters():
    #         param.requires_grad = False

    # Print trainable parameters
    trainable_params = model.num_parameters(only_trainable=True)
    total_params = model.num_parameters()
    accelerator.print(
        f"Model has {trainable_params:,} / {total_params:,} ({trainable_params / total_params:.2%}) trainable parameters"
    )

    ################
    # Dataset Preparation
    ################
    train_datasets = []
    test_datasets = []

    for dataset_path in script_args.dataset_paths:
        accelerator.print(f"Loading dataset from {dataset_path}...")
        dataset = load_from_disk(dataset_path)
        assert dataset.stage == "fsl", f"Expected FSL dataset, got {dataset.stage}"
        dataset.auto_set_transform(
            stage="fsl",
            load_images=True,
            image_processor=processor.image_processor,
            pad_token_id=tokenizer.pad_token_id,
        )

        if "train" in dataset:
            train_datasets.append(dataset["train"])
        if "test" in dataset:
            test_datasets.append(limit_dataset(dataset["test"]))

    train_fsl_dataset = ConcatDataset(train_datasets)
    eval_fsl_dataset = ConcatDataset(test_datasets) if test_datasets else None

    ################
    # Data Collator
    ################
    # Get the appropriate collate function based on model type
    collate_fn = get_collate_fn(model_args.model_name_or_path)
    data_collator = lambda examples: collate_fn(examples, script_args.max_sequence_length, processor)  # noqa: E731
    accelerator.print(f"Using collate function for model type: {model_type}")

    ################
    # Trainer
    ################
    accelerator.print("Initializing trainer...")
    # Type cast to suppress warnings - SFTTrainer accepts ConcatDataset
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_fsl_dataset,  # type: ignore
        eval_dataset=eval_fsl_dataset,  # type: ignore
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
