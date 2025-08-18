import argparse

import torch
from accelerate import Accelerator
from loguru import logger
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoProcessor

from owa.data.collator import detect_model_type, get_collate_fn
from owa.data.datasets import load_from_disk

# This line is to enable throughput logging from FSLTransform
# logger.enable("owa.data.datasets.transforms")


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return {
            "input_ids": torch.randint(0, 1024, (1024,), dtype=torch.long),
            "attention_mask": torch.randint(0, 1, (1024,), dtype=torch.long),
            "images": torch.rand(14, 3, 512, 512, dtype=torch.float32),
        }

    def __len__(self):
        return 1000000


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU FSL dataset loader")
    parser.add_argument(
        "datasets",
        nargs="+",
        help="List of dataset paths to load (e.g., /path/to/dataset1 /path/to/dataset2)",
    )
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        help="Model name for image processor (default: HuggingFaceTB/SmolVLM2-256M-Video-Instruct)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training (default: 8)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of data loader workers (default: 8)",
    )

    args = parser.parse_args()

    # 1) Initialize Accelerator
    accelerator = Accelerator()

    # 2) Load FSL datasets (pre-computed)
    print("▶ Loading FSL datasets…")
    train_datasets = []
    for dataset_path in args.datasets:
        logger.info(f"Loading dataset from: {dataset_path}")
        fsl_ds = load_from_disk(dataset_path)
        train_datasets.append(fsl_ds["train"])

    print("▶ Loading image processor…")
    # Detect model type for appropriate configuration
    model_type = detect_model_type(args.model)
    print(f"Detected model type: {model_type}")

    # Configure processor based on model type
    if model_type == "internvl3":
        # InternVL3 configuration: disable multi-crop for efficiency
        processor = AutoProcessor.from_pretrained(args.model)
        processor.image_processor = AutoImageProcessor.from_pretrained(
            args.model,
            use_fast=True,
            crop_to_patches=False,  # InternVL3-specific: disable multi-crop
        )
        print("Configured InternVL3 processor with multi-crop disabled")
    else:
        # SmolVLM2 and other models configuration
        processor = AutoProcessor.from_pretrained(args.model, do_image_splitting=False)
        processor.image_processor = AutoImageProcessor.from_pretrained(
            args.model,
            use_fast=True,
            do_image_splitting=False,  # SmolVLM2-specific: disable image splitting
        )
        print("Configured SmolVLM2/default processor")

    # 3) Apply FSL transform for on-the-fly processing and concatenate datasets
    for train_ds in train_datasets:
        train_ds.auto_set_transform(
            stage="fsl",
            load_images=True,
            image_processor=processor.image_processor,
            mcap_root_directory=None,
        )

    train_ds = ConcatDataset(train_datasets)
    # train_ds = DummyDataset()

    # 4) Create a DataLoader with appropriate collate function
    collate_fn_for_model = get_collate_fn(args.model)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=2,
        # persistent_workers=True,
        pin_memory=True,
        collate_fn=lambda examples: collate_fn_for_model(examples, max_sequence_length=1024, processor=processor),
    )
    print(f"Using collate function for model type: {model_type}")

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
