"""Event transform for OWA datasets."""

from typing import Optional

from mcap_owa.highlevel import McapMessage
from owa.data.encoders import create_encoder

from .utils import resolve_episode_path


def create_event_transform(
    encoder_type: str = "hierarchical", load_images: bool = True, mcap_root_directory: Optional[str] = None
):
    """Create transform for EVENT stage."""

    def transform_batch(batch):
        encoder = create_encoder(encoder_type)
        episode_paths = [resolve_episode_path(path, mcap_root_directory) for path in batch.get("episode_path", [])]
        results = {"encoded_event": [], "images": []}

        for i in range(len(batch["mcap_message"])):
            mcap_msg = McapMessage.model_validate_json(batch["mcap_message"][i].decode("utf-8"))
            encoded_text, screen_captured = encoder.encode(mcap_msg)

            images = []
            if batch["topic"][i] == "screen" and screen_captured and load_images:
                for screen in screen_captured:
                    screen.resolve_relative_path(episode_paths[i])
                    images.append(screen.to_pil_image(keep_av_open=True))

            results["encoded_event"].append(encoded_text)
            results["images"].append(images)

        return results

    return transform_batch
