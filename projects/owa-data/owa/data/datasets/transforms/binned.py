"""Binned transform for OWA datasets."""

from typing import Optional

from mcap_owa.highlevel import McapMessage
from owa.data.encoders import create_encoder

from .utils import resolve_episode_path


def create_binned_transform(
    instruction: str = "Complete the computer task",
    encoder_type: str = "hierarchical",
    load_images: bool = True,
    encode_actions: bool = True,
    mcap_root_directory: Optional[str] = None,
):
    """Create transform for BINNED stage."""

    def transform_batch(batch):
        encoder = create_encoder(encoder_type) if encode_actions else None
        episode_paths = [resolve_episode_path(path, mcap_root_directory) for path in batch.get("episode_path", [])]
        batch_size = len(batch[list(batch.keys())[0]])
        state, actions = [], []
        for i in range(batch_size):
            _state, _action = [], []
            for msg in batch["state"][i]:
                mcap_msg = McapMessage.model_validate_json(msg.decode("utf-8"))
                if mcap_msg.message_type == "desktop/ScreenCaptured":
                    screen = mcap_msg.decoded
                    screen.resolve_relative_path(episode_paths[i])
                screen_value = screen.to_pil_image(keep_av_open=True) if load_images else screen
                _state.append(screen_value)
            for msg in batch["actions"][i]:
                mcap_msg = McapMessage.model_validate_json(msg.decode("utf-8"))
                if encode_actions:
                    action, image = encoder.encode(mcap_msg)
                    assert len(image) == 0, "Action encoding should not produce images"
                    _action.append(action)
                else:
                    _action.append(mcap_msg)
            if encode_actions:
                _action = "".join(_action)
            state.append(_state)
            actions.append(_action)

        return {
            "instruction": [instruction] * batch_size,
            "state": state,
            "actions": actions,
        }

    return transform_batch
