"""
This script's I/O

Input: list[query]
Output: N/A
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

import cv2
import line_profiler
import pandas as pd
import torch
import typer
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, GPT2Tokenizer, GPT2TokenizerFast

from owa_game_agent.data import OWAMcapQuery
from owa_game_agent.data.datasets.smolvlm2 import SmolVLM2Dataset, collate_fn, sample_to_smolvlm_input
from owa_game_agent.data.sample_processor import (
    KEYBOARD_EVENT_TOKEN_FORMAT,
    KEYBOARD_STATE_COUNT,
    KEYBOARD_VK_COUNT,
    TIMESTAMP_TOKEN_COUNT,
    TIMESTAMP_TOKEN_FORMAT,
    SampleProcessor,
)

app = typer.Typer()


def load_queries(query_path: Path) -> List[OWAMcapQuery]:
    """Load queries from a JSONL file."""
    with open(query_path, "r") as f:
        return [OWAMcapQuery.model_validate_json(line) for line in f]


@app.command("quickstart")
def show_by_decompose(query_path: Path):
    """
    example output:

    state_keyboard=set() state_mouse={'pressed': set(), 'x': 1061, 'y': 812} state_screen=[(-227049220, {'path': '/mnt/raid11/datasets/owa/mcaps/ztype.mkv', 'pts': 208520000000, 'utc_ns': 1741776086345743100}), (-160120220, {'path': '/mnt/raid11/datasets/owa/mcaps/ztype.mkv', 'pts': 208590000000, 'utc_ns': 1741776086412672100}), (-110049820, {'path': '/mnt/raid11/datasets/owa/mcaps/ztype.mkv', 'pts': 208640000000, 'utc_ns': 1741776086462404300}), (-59651220, {'path': '/mnt/raid11/datasets/owa/mcaps/ztype.mkv', 'pts': 208690000000, 'utc_ns': 1741776086512804500}), (-10007820, {'path': '/mnt/raid11/datasets/owa/mcaps/ztype.mkv', 'pts': 208740000000, 'utc_ns': 1741776086562784500})] action_keyboard=[(9606680, {'event_type': 'release', 'vk': 83}), (100317080, {'event_type': 'press', 'vk': 76}), (187930380, {'event_type': 'press', 'vk': 69}), (233117480, {'event_type': 'release', 'vk': 76})] action_mouse=[]
    ==============
    state_keyboard=[] state_mouse={'pressed': set(), 'x': 1061, 'y': 812} state_screen=[(-227049220, {'path': '/mnt/raid11/datasets/owa/mcaps/ztype.mkv', 'pts': 208520000000, 'utc_ns': 1741776086345743100}), (-160120220, {'path': '/mnt/raid11/datasets/owa/mcaps/ztype.mkv', 'pts': 208590000000, 'utc_ns': 1741776086412672100}), (-110049820, {'path': '/mnt/raid11/datasets/owa/mcaps/ztype.mkv', 'pts': 208640000000, 'utc_ns': 1741776086462404300}), (-59651220, {'path': '/mnt/raid11/datasets/owa/mcaps/ztype.mkv', 'pts': 208690000000, 'utc_ns': 1741776086512804500}), (-10007820, {'path': '/mnt/raid11/datasets/owa/mcaps/ztype.mkv', 'pts': 208740000000, 'utc_ns': 1741776086562784500})] action_keyboard=['<TIMESTAMP_0>', '<KEYBOARD_83_0>', '<TIMESTAMP_10>', '<KEYBOARD_76_1>', '<TIMESTAMP_18>', '<KEYBOARD_69_1>', '<TIMESTAMP_23>', '<KEYBOARD_76_0>'] action_mouse=[]
    ==============
    images=[<PIL.Image.Image image mode=RGB size=794x794 at 0x7F3100962DD0>, <PIL.Image.Image image mode=RGB size=794x794 at 0x7F30114234D0>, <PIL.Image.Image image mode=RGB size=794x794 at 0x7F301142DF90>, <PIL.Image.Image image mode=RGB size=794x794 at 0x7F31009609D0>, <PIL.Image.Image image mode=RGB size=794x794 at 0x7F31009608D0>] messages=[{'role': 'user', 'content': [{'type': 'text', 'text': "You are playing Super Hexagon, a fast-paced game that requires precise control and timing. The current keyboard state, which represents the keys that are pressed, is .\nAfter this prompt, you will receive 5 sequential image frames that show the game's visual history from the past to the present.\nUsing the current keyboard state and the image sequence, predict the future sequence of keyboard actions. For each action, include the timestamp when it should be executed.<image><image><image><image><image>"}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': '<TIMESTAMP_0><KEYBOARD_83_0><TIMESTAMP_10><KEYBOARD_76_1><TIMESTAMP_18><KEYBOARD_69_1><TIMESTAMP_23><KEYBOARD_76_0>'}]}]
    ==============
    """
    # Load all queries from the file
    queries = load_queries(query_path)

    # Select a sample query for processing (same as original)
    query_index = len(queries) // 2 + 55
    query = queries[query_index]

    # Process the sample
    sample = query.to_sample()
    print(sample)
    print("==============")

    sample_processor = SampleProcessor()
    tokenized_sample = sample_processor.tokenize(sample)
    print(tokenized_sample)
    print("==============")

    vlm_input = sample_to_smolvlm_input(tokenized_sample)
    print(vlm_input)
    print("==============")


@app.command("eda")
def eda_sample(query_path: Path):
    # Load all queries from the file
    queries = load_queries(query_path)

    # Parallel process to convert queries to samples
    samples = []
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(query.to_sample) for query in queries]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing queries"):
            query = future.result()
            samples.append(query)

    action_keyboards = []
    for sample in samples:
        for timestamp, msg in sample.action_keyboard:
            action_keyboards.append((msg["vk"], msg["event_type"]))

    # show up the distribution of action_keyboard. I'm curious about the majority event
    df = pd.DataFrame(action_keyboards, columns=["vk", "event_type"])
    print(df["vk"].value_counts())
    print(df["event_type"].value_counts())
    # print major in terms of (vk, event_type) pairs
    print(df.groupby(["vk", "event_type"]).size().sort_values(ascending=False))

    # show up the distribution of sample.action_keyboard's length
    action_keyboard_lengths = [len(sample.action_keyboard) for sample in samples]

    df = pd.DataFrame(action_keyboard_lengths, columns=["action_keyboard_length"])
    print(df.describe())

    # print ratio of 0
    ratio_of_zero = action_keyboard_lengths.count(0) / len(action_keyboard_lengths)
    print(f"Ratio of 0: {ratio_of_zero}")
    if ratio_of_zero > 0.5:
        logger.warning("The dataset has many samples with 0 action_keyboard length.")

    # print example sample which has maximum action_keyboard length
    max_idx = action_keyboard_lengths.index(max(action_keyboard_lengths))
    print(samples[max_idx])


@app.command("prepare")
def prepare_model(
    save_path: Path, model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", apply_semantic_init: bool = False
):
    model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    # https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/discussions/16
    processor = AutoProcessor.from_pretrained(
        model_id,
        padding_side="left",
        do_image_splitting=False,
        image_size={"longest_edge": 512},
        max_image_size={"longest_edge": 512},
    )

    # print(processor.image_size)
    # print(processor.image_processor.size)
    # print(processor.image_processor.max_image_size)

    processor.image_processor.size = {"longest_edge": 512}

    assert processor.tokenizer.padding_side == "left"  # original: "right"
    assert processor.do_image_splitting is False  # original: True
    # original: {"longest_edge": 2048}
    processor.image_processor.size = {"longest_edge": 512}
    assert processor.image_size == processor.image_processor.size == {"longest_edge": 512}
    # original: {"longest_edge": 512}
    assert processor.image_processor.max_image_size == {"longest_edge": 512}

    before_token_count = len(processor.tokenizer)  # 49280
    print(f"Before token count: {before_token_count}")
    print(f"Note that processor.tokenizer.vocab_size is {processor.tokenizer.vocab_size}, which seems to be mistake.")

    tokens_to_expand = []
    for i in range(KEYBOARD_VK_COUNT):
        for j in range(KEYBOARD_STATE_COUNT):
            tokens_to_expand.append(KEYBOARD_EVENT_TOKEN_FORMAT.format(i, j))

    for i in range(TIMESTAMP_TOKEN_COUNT):
        tokens_to_expand.append(TIMESTAMP_TOKEN_FORMAT.format(i))

    processor.tokenizer.add_tokens(tokens_to_expand)
    model.resize_token_embeddings(len(processor.tokenizer))
    # Outputs:
    # > The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
    # > The new lm_head weights will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`

    after_token_count = len(processor.tokenizer)  # 49892
    print(f"After token count: {after_token_count}")

    assert after_token_count == before_token_count + KEYBOARD_VK_COUNT * KEYBOARD_STATE_COUNT + TIMESTAMP_TOKEN_COUNT

    if apply_semantic_init:
        # Apply semantic initialization
        for i in range(KEYBOARD_VK_COUNT):
            # Skip non-alphabet characters
            source_char = None
            if 0x41 <= i <= 0x5A:
                source_char = chr(i)
            elif 0x25 <= i <= 0x28:
                # left, up, right, down arrow
                source_char = {0x25: "left", 0x26: "up", 0x27: "right", 0x28: "down"}[i]

            if source_char is None:
                continue

            for j in range(KEYBOARD_STATE_COUNT):
                # Initialize the embeddings for <KEYBOARD_i_j> as (i - 0x41)th alphabet character
                NEW_TOKEN = KEYBOARD_EVENT_TOKEN_FORMAT.format(i, j)
                new_token_idx = processor.tokenizer.convert_tokens_to_ids(NEW_TOKEN)
                source_token_idx = processor.tokenizer.convert_tokens_to_ids(source_char)
                if new_token_idx == 0 or source_token_idx == 0:
                    logger.warning(
                        f"token: {NEW_TOKEN}, new_token_idx: {new_token_idx}, source_token_idx: {source_token_idx}"
                    )
                    continue
                model.get_input_embeddings().weight.data[new_token_idx] = model.get_input_embeddings().weight.data[
                    source_token_idx
                ]
                model.get_output_embeddings().weight.data[new_token_idx] = model.get_output_embeddings().weight.data[
                    source_token_idx
                ]

        for i in range(TIMESTAMP_TOKEN_COUNT):
            # Initialize the embeddings for <TIMESTAMP_i> as digit i
            NEW_TOKEN = TIMESTAMP_TOKEN_FORMAT.format(i)
            new_token_idx = processor.tokenizer.convert_tokens_to_ids(NEW_TOKEN)
            source_token_idx = processor.tokenizer.convert_tokens_to_ids(str(i))
            if new_token_idx == 0 or source_token_idx == 0:
                logger.warning(
                    f"token: {NEW_TOKEN}, new_token_idx: {new_token_idx}, source_token_idx: {source_token_idx}"
                )
                continue
            model.get_input_embeddings().weight.data[new_token_idx] = model.get_input_embeddings().weight.data[
                source_token_idx
            ]
            model.get_output_embeddings().weight.data[new_token_idx] = model.get_output_embeddings().weight.data[
                source_token_idx
            ]

    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)


@app.command("verify")
def verify_tokenizer(model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"):
    """
    Verify the tokenizer with additional tokens.
    Related issue: https://github.com/huggingface/tokenizers/issues/1544
    """
    processor = AutoProcessor.from_pretrained(model_id)

    print(processor.tokenizer.eos_token_id)
    print(processor.tokenizer.pad_token_id)

    assert isinstance(processor.tokenizer, GPT2TokenizerFast)

    def test_tokenizer(tokenizer):
        test_texts = [
            "!@#",
            "!@# ",
            "!@# <ACTION_1>",
            "!@# <ACTION_1> ",
            "!@# <ACTION_1> <ACTION_2>",
            "!@# <ACTION_1><ACTION_2>",
        ]
        print("=======")
        tokenized = []
        for text in test_texts:
            print(f"{text:30}", tokenizer(text))
            tokenized.append(tokenizer(text))
        return tokenized

    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    tokenizer.add_tokens([f"<ACTION_{idx}>" for idx in range(18)])
    A = test_tokenizer(tokenizer)

    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    tokenizer.add_tokens([f"<ACTION_{idx}>" for idx in range(18)], special_tokens=True)
    B = test_tokenizer(tokenizer)

    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    tokenizer.add_tokens([f"<ACTION_{idx}>" for idx in range(18)])
    C = test_tokenizer(tokenizer)

    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    tokenizer.add_tokens([f"<ACTION_{idx}>" for idx in range(18)], special_tokens=True)
    D = test_tokenizer(tokenizer)

    assert A == B == C == D


@app.command("collator")
@line_profiler.profile
def show_dataset_collator(query_path: Path, model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"):
    processor = AutoProcessor.from_pretrained(model_id)

    assert processor.tokenizer.padding_side == "left"  # original: "right"
    assert processor.do_image_splitting is False  # original: True
    # original: {"longest_edge": 2048}
    assert processor.image_size == processor.image_processor.size == {"longest_edge": 512}
    # original: {"longest_edge": 512}
    assert processor.image_processor.max_image_size == {"longest_edge": 512}

    dataset = SmolVLM2Dataset(query_path)

    samples = [dataset[0], dataset[len(dataset) // 2 + 55]]

    samples[0]["images"][0].save("example_sample.png")

    print(samples)
    batch = collate_fn(samples, processor)

    pixel_values = batch["pixel_values"]
    pixel_attention_mask = batch["pixel_attention_mask"]
    input_ids: torch.LongTensor = batch["input_ids"]
    """
    In [16]: a="<end_of_utterance>\nAssistant: <end_of_utterance>"

    In [17]: processor.tokenizer(a)
    Out[17]: {'input_ids': [49279, 198, 9519, 9531, 42, 216, 49279], 'attention_mask': [1, 1, 1, 1, 1, 1, 1]}
    """
    attention_mask = batch["attention_mask"]
    labels: torch.LongTensor = batch["labels"]

    # torch.Size([2, 65, 3, 512, 512]) torch.Size([2, 65, 512, 512]) torch.Size([2, 4419]) torch.Size([2, 4419]) torch.Size([2, 4419])
    # torch.Size([2, 5, 3, 512, 512]) torch.Size([2, 5, 512, 512]) torch.Size([2, 439]) torch.Size([2, 439]) torch.Size([2, 439])
    print(pixel_values.shape, pixel_attention_mask.shape, input_ids.shape, attention_mask.shape, labels.shape)
    print(input_ids.tolist(), labels.tolist())

    # visualize pixel_values
    pixel_values = pixel_values[0]
    pixel_values = pixel_values.permute(0, 2, 3, 1)
    pixel_values = pixel_values.cpu().numpy()

    print(pixel_values.min(), pixel_values.max())  # -1, 1
    pixel_values = (pixel_values + 1) / 2 * 255
    pixel_values = pixel_values.astype("uint8")

    for i, image in enumerate(pixel_values):
        cv2.imwrite(f"image_{i}.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    app()
