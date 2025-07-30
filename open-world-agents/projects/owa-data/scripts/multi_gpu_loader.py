import line_profiler
import torch
from accelerate import Accelerator
from datasets import load_from_disk
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoProcessor

from owa.data.datasets import FSLDataset
from owa.data.episode_tokenizer import EpisodeTokenizer

logger.enable("owa.data.datasets.fsl_dataset")


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return {
            "input_ids": torch.randint(0, 1024, (1024,), dtype=torch.long),
            "attention_mask": torch.randint(0, 1, (1024,), dtype=torch.long),
            "images": torch.rand(14, 3, 512, 512, dtype=torch.float32),
        }

    def __len__(self):
        return 1000000


@line_profiler.profile
def collate_fn(examples, max_sequence_length: int | None = None, tokenizer=None):
    input_ids_list = []
    attention_mask_list = []
    pixel_values_list = []

    for example in examples:
        input_ids_list.append(example["input_ids"])  # [seq_len,]
        attention_mask_list.append(example["attention_mask"])  # [seq_len,]
        pixel_values_list.append(example["images"])  # [num_images, channels, height, width]
        assert isinstance(example["images"], torch.Tensor), f"Expected tensor, got {type(example['images'])}"

    max_num_images = max([len(images) for images in pixel_values_list])
    # Pad images to max_num_images
    for idx, images in enumerate(pixel_values_list):
        if len(images) < max_num_images:
            # NOTE: Idefics3/SmolVLM expect all-zero image to be a padding image. see: https://github.com/huggingface/transformers/blob/69b158260fcb679ea3bfbc1e6a358545ee53ee28/src/transformers/models/idefics3/modeling_idefics3.py#L693
            padding = torch.zeros(max_num_images - len(images), *images.shape[1:], dtype=torch.float32)
            pixel_values_list[idx] = torch.concat([images, padding])

    # Convert to tensors
    input_ids = torch.stack(input_ids_list)  # [batch_size, seq_len]
    attention_mask = torch.stack(attention_mask_list)  # [batch_size, seq_len]
    pixel_values = torch.stack(pixel_values_list)  # [batch_size, max_num_images, 3, max_heights, max_widths]

    if max_sequence_length is not None and input_ids.shape[1] != max_sequence_length:
        raise ValueError(
            f"Input ids length ({input_ids.shape[1]}) does not match max_sequence_length ({max_sequence_length})"
        )

    # NOTE: we shift the labels inside the model, so we don't need to do it here
    labels = input_ids.clone()
    if tokenizer is not None:
        # Ignore padding tokens in the loss computation
        labels[labels == tokenizer.pad_token_id] = -100
        # Ignore the image token index in the loss computation
        labels[labels == tokenizer.image_token_id] = -100
        assert (labels[attention_mask == 0] == -100).all()
    else:
        labels[attention_mask == 0] = -100

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
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
    # train_ds = DummyDataset()

    # 4) Create a DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=8,
        # persistent_workers=True,
        pin_memory=True,
        collate_fn=lambda examples: collate_fn(examples, max_sequence_length=1024, tokenizer=processor.tokenizer),
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

            pbar.update()  # expected: 1.5 s/it
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})


if __name__ == "__main__":
    main()
