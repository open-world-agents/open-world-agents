"""
Simple test for the EventEncoder implementations.

This module tests the basic functionality of the EventEncoder implementations,
particularly focusing on the image token prefix and suffix handling.
"""

import json

from mcap_owa.highlevel.reader import McapMessage
from owa.data.encoders import HierarchicalEventEncoder, JSONEventEncoder


def test_encoder_creation():
    """Test that encoders can be created and have the expected attributes."""
    # Test hierarchical encoder
    hierarchical_encoder = HierarchicalEventEncoder()
    assert hasattr(hierarchical_encoder, "config")
    print(f"Hierarchical encoder config: {hierarchical_encoder.config}")
    print(f"Has image_token_prefix: {hasattr(hierarchical_encoder.config, 'image_token_prefix')}")
    print(f"Has image_token_suffix: {hasattr(hierarchical_encoder.config, 'image_token_suffix')}")

    # Test JSON encoder
    json_encoder = JSONEventEncoder()
    assert hasattr(json_encoder, "config")
    print(f"JSON encoder config: {json_encoder.config}")
    print(f"Has image_token_prefix: {hasattr(json_encoder.config, 'image_token_prefix')}")
    print(f"Has image_token_suffix: {hasattr(json_encoder.config, 'image_token_suffix')}")

    # Test base config directly
    import inspect

    from owa.data.encoders.base_encoder import BaseEventEncoderConfig

    print(f"BaseEventEncoderConfig file: {inspect.getfile(BaseEventEncoderConfig)}")
    print(f"BaseEventEncoderConfig fields: {BaseEventEncoderConfig.__dataclass_fields__.keys()}")

    base_config = BaseEventEncoderConfig()
    print(f"Base config: {base_config}")
    print(f"Base has image_token_prefix: {hasattr(base_config, 'image_token_prefix')}")
    print(f"Base has image_token_suffix: {hasattr(base_config, 'image_token_suffix')}")

    # Test hierarchical config directly
    from owa.data.encoders.hierarchical_event_encoder import HierarchicalEventEncoderConfig

    hier_config = HierarchicalEventEncoderConfig()
    print(f"Hierarchical config: {hier_config}")
    print(f"Hierarchical has image_token_prefix: {hasattr(hier_config, 'image_token_prefix')}")
    print(f"Hierarchical has image_token_suffix: {hasattr(hier_config, 'image_token_suffix')}")


def test_hierarchical_encoder_screen_event():
    """Test that HierarchicalEventEncoder handles screen events with prefix/suffix."""
    encoder = HierarchicalEventEncoder()

    # Create a simple screen event message with media_ref to satisfy validation
    screen_data = {
        "utc_ns": 1000000000,
        "shape": [200, 100],
        "source_shape": [200, 100],
        "media_ref": {"uri": "test.png"},
    }
    mcap_message = McapMessage(
        topic="screen",
        timestamp=1000000000,
        message_type="desktop/ScreenCaptured",
        message=json.dumps(screen_data).encode("utf-8"),
    )

    # Test encoding
    encoded_text, images = encoder.encode(mcap_message)
    print(f"Encoded text: {encoded_text}")

    # Check that the encoded text contains the expected tokens
    assert encoder.config.image_token_prefix in encoded_text
    assert encoder.config.image_token in encoded_text
    assert encoder.config.image_token_suffix in encoded_text


def test_json_encoder_screen_event():
    """Test that JSONEventEncoder handles screen events with prefix/suffix."""
    encoder = JSONEventEncoder()

    # Create a simple screen event message with a mock decoded attribute
    screen_data = {
        "utc_ns": 1000000000,
        "shape": [200, 100],
        "source_shape": [200, 100],
        "media_ref": {"uri": "test.png"},
    }
    mcap_message = McapMessage(
        topic="screen",
        timestamp=1000000000,
        message_type="desktop/ScreenCaptured",
        message=json.dumps(screen_data).encode("utf-8"),
    )

    # Mock the decoded attribute for testing
    import numpy as np

    from owa.msgs.desktop.screen import ScreenCaptured

    # Create a frame array and a media reference to satisfy validation
    frame = np.zeros((100, 200, 4), dtype=np.uint8)
    mock_screen = ScreenCaptured(
        utc_ns=1000000000, frame_arr=frame, shape=(200, 100), source_shape=(200, 100), media_ref={"uri": "test.png"}
    )
    mcap_message.decoded = mock_screen

    # Test encoding
    encoded_text, images = encoder.encode(mcap_message)
    print(f"Encoded text: {encoded_text}")

    # Check that the encoded text contains the expected tokens
    print(f"Image token prefix: {encoder.config.image_token_prefix}")
    print(f"Image token: {encoder.config.image_token}")
    print(f"Image token suffix: {encoder.config.image_token_suffix}")

    # The token combination should be in the encoded text
    expected_token = (
        f"{encoder.config.image_token_prefix}{encoder.config.image_token}{encoder.config.image_token_suffix}"
    )
    print(f"Expected token: {expected_token}")
    assert expected_token in encoded_text


def test_encoder_vocab():
    """Test that encoder vocabularies include the image tokens."""
    hierarchical_encoder = HierarchicalEventEncoder()
    json_encoder = JSONEventEncoder()

    # Check hierarchical encoder vocab
    vocab = hierarchical_encoder.get_vocab()
    assert hierarchical_encoder.config.image_token in vocab
    assert hierarchical_encoder.config.image_token_prefix in vocab
    assert hierarchical_encoder.config.image_token_suffix in vocab

    # Check JSON encoder vocab
    vocab = json_encoder.get_vocab()
    assert json_encoder.config.image_token in vocab
    assert json_encoder.config.image_token_prefix in vocab
    assert json_encoder.config.image_token_suffix in vocab


if __name__ == "__main__":
    # Run the tests
    test_encoder_creation()
    test_hierarchical_encoder_screen_event()
    test_json_encoder_screen_event()
    test_encoder_vocab()
    print("All tests passed!")
