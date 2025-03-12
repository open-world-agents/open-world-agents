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
"""

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from trl import SFTTrainer, SFTConfig

CHAT_INSTRUCTION = """You are playing Super Hexagon, a fast-paced game that requires precise control and timing. The current keyboard state, which represents the keys that are pressed, is {keyboard_state}.
After this prompt, you will receive {len_images} sequential image frames that show the game’s visual history from the past to the present.
Using the current keyboard state and the image sequence, predict the future sequence of keyboard actions. For each action, include the timestamp when it should be executed."""

if __name__ == "__main__":
    model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    
    ##########
    # Model, Tokenizer & Processor
    ##########
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    )

    ################
    # Create a data collator to encode text and image pairs
    ################
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        texts = []
        images = []
        for example in examples:
            keyboard_state = example["state_keyboard"]
            keyboard_action = example["action_keyboard"]
            state_screen = example["state_screen"]

            len_images = len(state_screen)

            # It will be processed like [keyboard1, keyboard2, ...], -> keyboard1keyboard2...
            keyboard_state = "".join(str(item) for item in keyboard_state)
            # It will be processed like [(timestamp1, keyboard1), (timestamp2, keyboard2), ...], -> timestamp1keyboard1timestamp2keyboard2...
            keyboard_action = "".join(str(item) for pair in keyboard_action for item in pair)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": CHAT_INSTRUCTION.format_map({"len_images": len_images, "keyboard_state": keyboard_state})}+"<image>"*len_images,
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": keyboard_action},
                    ]
                }
            ]

            texts.append(processor.apply_chat_template(messages, tokenize=False))
            images.append(state_screen)

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        # Ignore the image token index in the loss computation (model specific, we will use SmolVLM2)
        image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        # TODO: ignore the prompt token index if needed

        return batch

    ################
    # Dataset
    ################
    dataset = # TODO: Huggingface dataset?

    ################
    # Training arguments
    ################

    save_path = "/mnt/cephfs/scratch"

    args = SFTConfig(
        # deepspeed=deepspeed_conf,
        output_dir=save_path,  # directory to save and repository id
        num_train_epochs=1,                     # number of training epochs
        per_device_train_batch_size=1,          # batch size per device during training
        gradient_accumulation_steps=4,         # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=1,                        # log every step
        save_strategy="steps",                  # save checkpoint every epoch
        save_steps=2000,
        save_total_limit=1,
        learning_rate=2e-5,                     # learning rate, based on QLoRA paper
        bf16=True,                              # use bfloat16 precision
        tf32=True,                              # use tf32 precision
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        weight_decay=0.,
        lr_scheduler_type="cosine",
        max_seq_length=6144,
        report_to="wandb",
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        dataset_text_field="", # need a dummy field for collator
        dataset_kwargs = {"skip_prepare_dataset": True}, # important for collator
        dataloader_num_workers=4,
        run_name=save_path.split('/')[-1],
    )

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=None,
        processing_class=processor.tokenizer,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(save_path)