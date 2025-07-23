import line_profiler
import torch
from accelerate import Accelerator
from datasets import load_from_disk
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoProcessor

from owa.data.episode_tokenizer import EpisodeTokenizer
from owa.data.fsl_dataset import FSLDataset

logger.enable("owa.data.fsl_dataset")


@line_profiler.profile
def collate_fn(examples):
    # batch = {
    #     "input_ids": torch.randint(0, 1000, (1, 1024), dtype=torch.long),
    #     "attention_mask": torch.randint(0, 1, (1, 1024), dtype=torch.long),
    #     "labels": torch.randint(0, 1000, (1, 1024), dtype=torch.long),
    #     "image_hidden_states": torch.rand(112, 3, 512, 512, dtype=torch.float32),
    # }
    # return batch

    input_ids_list = []
    attention_mask_list = []
    image_hidden_states_list = []

    for example in examples:
        input_ids_list.append(example["input_ids"])  # [seq_len,]
        attention_mask_list.append(example["attention_mask"])  # [seq_len,]
        image_hidden_states_list.append(example["images"]["pixel_values"])  # [num_images, channels, height, width]
        print(example["images"]["pixel_values"].shape, example["images"]["pixel_values"].dtype)

    # Convert to tensors
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)  # [batch_size, seq_len]
    attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)  # [batch_size, seq_len]
    image_hidden_states = torch.concat(image_hidden_states_list)  # [total_num_images, channels, height, width]

    # For pretraining, labels are the same as input_ids (next token prediction)
    # We shift the labels inside the model, so we don't need to do it here
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "image_hidden_states": image_hidden_states,
    }
    return batch


def main():
    # 1) Initialize Accelerator
    accelerator = Accelerator()

    # 2) Load & tokenize your event dataset
    print("▶ Loading raw dataset…")
    event_ds = load_from_disk("/mnt/raid12/datasets/owa/data/super-hexagon-event")

    print("▶ Loading tokenizer…")
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct", do_image_splitting=False)
    processor.image_processor = AutoImageProcessor.from_pretrained(
        "HuggingFaceTB/SmolVLM2-256M-Video-Instruct", use_fast=True, do_image_splitting=False
    )

    print("▶ Preparing EpisodeTokenizer…")
    ep_tok = EpisodeTokenizer(image_token="<image>")
    ep_tok.prepare_model(tokenizer=processor.tokenizer)

    print("▶ Tokenizing splits…")
    tokenized = {}
    for split, ds in event_ds.items():
        tokenized[split] = ep_tok.tokenize_event_dataset(ds)

    # 3) Wrap into your FSLDataset (only train shown here)
    train_ds = FSLDataset(
        tokenized["train"],
        image_processor=processor.image_processor,
        pad_token_id=processor.tokenizer.pad_token_id,
        max_sequence_length=1024,
    )
    train_ds.prepare()

    # 4) Create a DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        # persistent_workers=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # 5) (Optional) A dummy model so you can do a full prepare()
    model = torch.nn.Linear(1024, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 6) Let Accelerator wrap model, optimizer, and dataloader
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # 7) Simple loop to verify each GPU/process sees its shard
    pbar = tqdm(total=2 * len(train_loader), disable=not accelerator.is_local_main_process)
    for epoch in range(2):
        for step, batch in enumerate(train_loader):
            # batch["input_ids"] is on the correct device
            # (B, seq_len) → just do a dummy forward
            loss = model(batch["input_ids"].float()).mean()
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            pbar.update()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})


if __name__ == "__main__":
    main()
