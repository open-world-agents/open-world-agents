from pathlib import Path

import torch
import typer
from loguru import logger
from transformers import AutoModelForImageTextToText, AutoProcessor, GPT2Tokenizer, GPT2TokenizerFast

from owa.agent.systems.goat.utils import EventProcessorConfig

app = typer.Typer()


@app.command()
def prepare_model(
    save_path: Path,
    model_id: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    apply_semantic_init: bool = True,
):
    # TODO: make this configurable
    event_config = EventProcessorConfig()

    # Load base model and processor
    model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    # https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/discussions/16
    processor = AutoProcessor.from_pretrained(
        model_id,
        padding_side="left",
        do_image_splitting=False,
        image_size={"longest_edge": 512},
        max_image_size={"longest_edge": 512},
    )

    # Original processor settings - leave correct, or fix if they're not.
    processor.image_processor.size = {"longest_edge": 512}

    assert processor.tokenizer.padding_side == "left"  # original: "right"
    assert processor.do_image_splitting is False  # original: True
    processor.image_processor.size = {"longest_edge": 512}
    assert processor.image_size == processor.image_processor.size == {"longest_edge": 512}
    assert processor.image_processor.max_image_size == {"longest_edge": 512}

    before_token_count = len(processor.tokenizer)
    print(f"Before token count: {before_token_count}")
    print(f"Note that processor.tokenizer.vocab_size is {processor.tokenizer.vocab_size}, which seems to be mistake.")

    # ===============================
    # Expand tokenizer with event tokens from config
    # ===============================

    # Aggregate all new event tokens
    tokens_to_expand = []
    tokens_to_expand.extend(event_config.keyboard_tokens)
    tokens_to_expand.extend(event_config.timestamp_tokens)
    tokens_to_expand.extend(event_config.mouse_move_tokens)
    tokens_to_expand.extend(event_config.mouse_click_tokens)
    tokens_to_expand.extend(event_config.mouse_scroll_tokens)
    tokens_to_expand.extend(event_config.screen_tokens)

    assert processor.tokenizer.add_tokens(tokens_to_expand) == len(tokens_to_expand)
    model.resize_token_embeddings(len(processor.tokenizer))
    # Outputs:
    # > The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
    # > The new lm_head weights will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`

    after_token_count = len(processor.tokenizer)
    print(f"After token count: {after_token_count}. Note that <image> was already in the tokenizer.")

    # ===============================
    # Semantic embedding initialization via heuristics
    # ===============================
    if apply_semantic_init:
        # --- Keyboard tokens: initialize e.g. <KEYBOARD_65_press> to the "A" embedding, arrow to corresponding word, etc.
        for vk in range(event_config.keyboard_vk_count):
            # You may encode heuristics such as letters/arrows
            source_char = None
            if 0x41 <= vk <= 0x5A:
                source_char = chr(vk)
            elif 0x25 <= vk <= 0x28:
                # left, up, right, down arrow
                source_char = {0x25: "left", 0x26: "up", 0x27: "right", 0x28: "down"}[vk]

            if source_char is None:
                continue

            NEW_TOKEN_PRESS = event_config.keyboard_token_format.format(vk=vk, pressed="press")
            NEW_TOKEN_RELEASE = event_config.keyboard_token_format.format(vk=vk, pressed="release")
            # May use same source_char for both press and release
            for NEW_TOKEN in [NEW_TOKEN_PRESS, NEW_TOKEN_RELEASE]:
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
        # --- Timestamp tokens: initialize e.g. <TIMESTAMP_0> to "0", etc.
        for idx, ts_token in enumerate(event_config.timestamp_tokens):
            new_token_idx = processor.tokenizer.convert_tokens_to_ids(ts_token)
            digit_str = str(idx)
            # Use digit matching only for 0-9, else skip or adjust as needed
            if 0 <= idx <= 9:
                source_token_idx = processor.tokenizer.convert_tokens_to_ids(digit_str)
                if new_token_idx == 0 or source_token_idx == 0:
                    logger.warning(
                        f"token: {ts_token}, new_token_idx: {new_token_idx}, source_token_idx: {source_token_idx}"
                    )
                    continue
                model.get_input_embeddings().weight.data[new_token_idx] = model.get_input_embeddings().weight.data[
                    source_token_idx
                ]
                model.get_output_embeddings().weight.data[new_token_idx] = model.get_output_embeddings().weight.data[
                    source_token_idx
                ]
        # (Optionally, you may add heuristic inits for mouse tokens, e.g. <MOUSE_click_left_press> using "left" + "press". Not done here.)

    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)


@app.command("verify")
def verify_tokenizer(model_id: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruc"):
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


if __name__ == "__main__":
    app()
