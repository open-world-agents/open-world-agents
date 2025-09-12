"""
Semantic initialization for hierarchical event tokens.

This module provides functionality to initialize token embeddings with semantic meanings
by copying embeddings from related natural language tokens.

All initialization uses the mean() method for consistency across different tokenizers:
- Single tokens: mean([x]) = x (same as direct indexing)
- Multiple tokens: mean([x,y,z]) = average (better semantic representation)
"""

from typing import Any, Dict

from loguru import logger
from transformers.tokenization_utils import PreTrainedTokenizer


def apply_semantic_initialization(tokenizer: PreTrainedTokenizer, model: Any, encoder_type: str) -> None:
    """
    Apply semantic initialization to hierarchical event tokens.

    This function initializes special tokens with embeddings from semantically related
    natural language tokens to provide better starting representations for training.

    Args:
        tokenizer: The tokenizer containing the special tokens
        model: The model with embedding weights to initialize
        encoder_type: The type of encoder used in the tokenizer
    """
    logger.warning("Applying semantic initialization to new tokens")

    # find out if tokenizer is expanded by hierarchical or factorized
    # if factorized, initialize factorized tokens
    # if hierarchical, initialize hierarchical tokens
    if encoder_type == "hierarchical":
        # Initialize basic numeric and semantic tokens
        _init_hierarchical_basic_tokens(tokenizer, model)
    elif encoder_type == "factorized":
        # Initialize basic numeric and semantic tokens
        _init_factorized_basic_tokens(tokenizer, model)

        # Initialize keyboard VK tokens
        _init_keyboard_vk_tokens(tokenizer, model)

        # Initialize mouse button tokens
        _init_mouse_button_tokens(tokenizer, model)
    else:
        raise ValueError(f"Invalid encoder type: {encoder_type}")


def _init_hierarchical_basic_tokens(tokenizer: PreTrainedTokenizer, model: Any) -> None:
    # add semantic init <0> ~ <15> tokens from 0 ~ 15
    for i in range(16):
        new_token_idx = tokenizer.convert_tokens_to_ids(f"<{i}>")
        if new_token_idx != tokenizer.unk_token_id:
            try:
                semantic_token_ids = tokenizer(str(i), add_special_tokens=False)["input_ids"]
                if semantic_token_ids:  # Only if semantic text was tokenized successfully
                    semantic_embedding = model.get_input_embeddings().weight.data[semantic_token_ids].mean(dim=0)
                    model.get_input_embeddings().weight.data[new_token_idx] = semantic_embedding
            except Exception:
                logger.warning(f"Failed to initialize token <{i}>")
                pass

    # Initialize semantic tokens from their natural language equivalents
    semantic_mappings = {
        "<KEYBOARD>": "keyboard",
        "<MOUSE>": "mouse",
        "<SCREEN>": "screen",
        "<press>": "press",
        "<release>": "release",
        "<EVENT_START>": "event start",
        "<EVENT_END>": "event end",
    }

    for new_token, semantic_word in semantic_mappings.items():
        new_token_idx = tokenizer.convert_tokens_to_ids(new_token)
        if new_token_idx != tokenizer.unk_token_id:
            try:
                semantic_token_ids = tokenizer(semantic_word, add_special_tokens=False)["input_ids"]
                if semantic_token_ids:  # Only if semantic text was tokenized successfully
                    semantic_embedding = model.get_input_embeddings().weight.data[semantic_token_ids].mean(dim=0)
                    model.get_input_embeddings().weight.data[new_token_idx] = semantic_embedding
            except Exception:
                logger.warning(f"Failed to initialize token {new_token}")
                pass


def _init_factorized_basic_tokens(tokenizer: PreTrainedTokenizer, model: Any) -> None:
    """Initialize basic numeric and semantic tokens using mean() method for consistency."""

    # Initialize numeric tokens <0> ~ <9> from their string representations
    for i in range(10):
        new_token_idx = tokenizer.convert_tokens_to_ids(f"<{i}>")
        if new_token_idx != tokenizer.unk_token_id:
            try:
                semantic_token_ids = tokenizer(str(i), add_special_tokens=False)["input_ids"]
                if semantic_token_ids:  # Only if semantic text was tokenized successfully
                    semantic_embedding = model.get_input_embeddings().weight.data[semantic_token_ids].mean(dim=0)
                    model.get_input_embeddings().weight.data[new_token_idx] = semantic_embedding
            except Exception:
                logger.warning(f"Failed to initialize token <{i}>")
                pass

    # Initialize semantic tokens from their natural language equivalents
    semantic_mappings = {
        "<KEYBOARD>": "keyboard",
        "<MOUSE>": "mouse",
        "<SCREEN>": "screen",
        "<press>": "press",
        "<release>": "release",
        "<EVENT_START>": "event start",
        "<EVENT_END>": "event end",
        "<SIGN_PLUS>": "+",
        "<SIGN_MINUS>": "-",
    }

    for new_token, semantic_word in semantic_mappings.items():
        new_token_idx = tokenizer.convert_tokens_to_ids(new_token)
        if new_token_idx != tokenizer.unk_token_id:
            try:
                semantic_token_ids = tokenizer(semantic_word, add_special_tokens=False)["input_ids"]
                if semantic_token_ids:  # Only if semantic text was tokenized successfully
                    semantic_embedding = model.get_input_embeddings().weight.data[semantic_token_ids].mean(dim=0)
                    model.get_input_embeddings().weight.data[new_token_idx] = semantic_embedding
            except Exception:
                logger.warning(f"Failed to initialize token {new_token}")
                pass


def _init_keyboard_vk_tokens(tokenizer: PreTrainedTokenizer, model: Any) -> None:
    """Initialize keyboard VK tokens with semantic embeddings."""

    # Create mapping of VK codes to semantic descriptions
    vk_semantics = _get_vk_semantics_mapping()

    # Initialize VK tokens with semantic embeddings
    for vk_code, semantic_text in vk_semantics.items():
        vk_token = f"<VK_{vk_code}>"
        vk_token_idx = tokenizer.convert_tokens_to_ids(vk_token)

        if vk_token_idx != tokenizer.unk_token_id:  # Only if token exists in vocab
            try:
                semantic_token_ids = tokenizer(semantic_text, add_special_tokens=False)["input_ids"]
                if semantic_token_ids:  # Only if semantic text was tokenized successfully
                    semantic_embedding = model.get_input_embeddings().weight.data[semantic_token_ids].mean(dim=0)
                    model.get_input_embeddings().weight.data[vk_token_idx] = semantic_embedding
            except Exception:
                logger.warning(f"Failed to initialize token {vk_token}")
                pass


def _get_vk_semantics_mapping() -> Dict[int, str]:
    """Get mapping of VK codes to semantic descriptions."""
    from owa.env.desktop.constants import VK

    return {
        # Letter keys
        VK.KEY_A: "A",
        VK.KEY_B: "B",
        VK.KEY_C: "C",
        VK.KEY_D: "D",
        VK.KEY_E: "E",
        VK.KEY_F: "F",
        VK.KEY_G: "G",
        VK.KEY_H: "H",
        VK.KEY_I: "I",
        VK.KEY_J: "J",
        VK.KEY_K: "K",
        VK.KEY_L: "L",
        VK.KEY_M: "M",
        VK.KEY_N: "N",
        VK.KEY_O: "O",
        VK.KEY_P: "P",
        VK.KEY_Q: "Q",
        VK.KEY_R: "R",
        VK.KEY_S: "S",
        VK.KEY_T: "T",
        VK.KEY_U: "U",
        VK.KEY_V: "V",
        VK.KEY_W: "W",
        VK.KEY_X: "X",
        VK.KEY_Y: "Y",
        VK.KEY_Z: "Z",
        # Number keys
        VK.KEY_0: "0",
        VK.KEY_1: "1",
        VK.KEY_2: "2",
        VK.KEY_3: "3",
        VK.KEY_4: "4",
        VK.KEY_5: "5",
        VK.KEY_6: "6",
        VK.KEY_7: "7",
        VK.KEY_8: "8",
        VK.KEY_9: "9",
        # Special keys
        VK.TAB: "TAB",
        VK.RETURN: "ENTER",
        VK.SPACE: "SPACE",
        VK.SHIFT: "SHIFT",
        VK.CONTROL: "CTRL",
        VK.MENU: "ALT",
        VK.ESCAPE: "ESC",
        VK.BACK: "BACKSPACE",
        VK.DELETE: "DELETE",
        VK.INSERT: "INSERT",
        VK.HOME: "HOME",
        VK.END: "END",
        VK.PRIOR: "PAGE UP",
        VK.NEXT: "PAGE DOWN",
        # Arrow keys
        VK.LEFT: "LEFT ARROW",
        VK.RIGHT: "RIGHT ARROW",
        VK.UP: "UP ARROW",
        VK.DOWN: "DOWN ARROW",
        # Function keys
        VK.F1: "F1",
        VK.F2: "F2",
        VK.F3: "F3",
        VK.F4: "F4",
        VK.F5: "F5",
        VK.F6: "F6",
        VK.F7: "F7",
        VK.F8: "F8",
        VK.F9: "F9",
        VK.F10: "F10",
        VK.F11: "F11",
        VK.F12: "F12",
        # Numpad
        VK.NUMPAD0: "NUMPAD 0",
        VK.NUMPAD1: "NUMPAD 1",
        VK.NUMPAD2: "NUMPAD 2",
        VK.NUMPAD3: "NUMPAD 3",
        VK.NUMPAD4: "NUMPAD 4",
        VK.NUMPAD5: "NUMPAD 5",
        VK.NUMPAD6: "NUMPAD 6",
        VK.NUMPAD7: "NUMPAD 7",
        VK.NUMPAD8: "NUMPAD 8",
        VK.NUMPAD9: "NUMPAD 9",
        VK.MULTIPLY: "NUMPAD MULTIPLY",
        VK.ADD: "NUMPAD PLUS",
        VK.SUBTRACT: "NUMPAD MINUS",
        VK.DECIMAL: "NUMPAD DECIMAL",
        VK.DIVIDE: "NUMPAD DIVIDE",
        # Windows keys
        VK.LWIN: "LEFT WINDOWS",
        VK.RWIN: "RIGHT WINDOWS",
        VK.APPS: "MENU",
    }


def _init_mouse_button_tokens(tokenizer: PreTrainedTokenizer, model: Any) -> None:
    """Initialize mouse button tokens with semantic embeddings."""
    # Create mapping of mouse button flag hex digits to semantic descriptions
    # Mouse button flags are encoded as 3-digit hex, each digit (0-15) gets its own token
    button_semantics = {
        # Common button flag patterns (hex digits 0-15)
        0: "no button",
        1: "left button down",
        2: "left button up",
        3: "button combination",
        4: "right button down",
        5: "button combination",
        6: "button combination",
        7: "button combination",
        8: "right button up",
        9: "button combination",
        10: "button combination",  # 0xA
        11: "button combination",  # 0xB
        12: "button combination",  # 0xC
        13: "button combination",  # 0xD
        14: "button combination",  # 0xE
        15: "button combination",  # 0xF
    }

    # Initialize mouse button tokens with semantic embeddings
    for hex_digit, semantic_text in button_semantics.items():
        mb_token = f"<MB_{hex_digit}>"
        mb_token_idx = tokenizer.convert_tokens_to_ids(mb_token)

        if mb_token_idx != tokenizer.unk_token_id:  # Only if token exists in vocab
            try:
                semantic_token_ids = tokenizer(semantic_text, add_special_tokens=False)["input_ids"]
                if semantic_token_ids:  # Only if semantic text was tokenized successfully
                    semantic_embedding = model.get_input_embeddings().weight.data[semantic_token_ids].mean(dim=0)
                    model.get_input_embeddings().weight.data[mb_token_idx] = semantic_embedding
            except Exception:
                logger.warning(f"Failed to initialize token {mb_token}")
                pass
