"""
Comprehensive tests for decode_args functionality in OWAMcapReader.

Tests all decode_args configurations with minimal code and maximum coverage.
"""

import warnings

import pytest
from easydict import EasyDict

from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter
from mcap_owa.types import DecodeArgs
from owa.core.message import OWAMessage


class KnownMessage(OWAMessage):
    """Message type that might be in registry (but actually isn't in test env)."""

    _type = "test/KnownMessage"
    data: str


class UnknownMessage(OWAMessage):
    """Message type that definitely won't be in registry."""

    _type = "unknown/TestMessage"
    value: int


@pytest.fixture
def mcap_file_with_mixed_messages(tmp_path):
    """Create MCAP file with both known and unknown message types."""
    file_path = tmp_path / "mixed.mcap"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        with OWAMcapWriter(file_path) as writer:
            # Known message (will decode successfully)
            writer.write_message(KnownMessage(data="test"), topic="/known", timestamp=1000)
            # Unknown message (will fail typed decoding)
            writer.write_message(UnknownMessage(value=42), topic="/unknown", timestamp=2000)

    return str(file_path)


@pytest.mark.parametrize(
    "decode_args,expected_types,should_fail",
    [
        # Typed decoding (default) - fails on unknown types (both are unknown in test env)
        ({}, [None, None], True),
        ({"return_dict": False, "return_dict_on_failure": False}, [None, None], True),
        # Dictionary decoding - always works, always returns EasyDict
        ({"return_dict": True}, [EasyDict, EasyDict], False),
        # Robust decoding - both fall back to dict since both are unknown in test env
        ({"return_dict_on_failure": True}, [EasyDict, EasyDict], False),
        ({"return_dict": False, "return_dict_on_failure": True}, [EasyDict, EasyDict], False),
    ],
)
def test_decode_args_configurations(mcap_file_with_mixed_messages, decode_args, expected_types, should_fail):
    """Test all decode_args configurations with both known and unknown message types."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Domain-based message.*not found in registry.*", UserWarning)
        warnings.filterwarnings("ignore", "Reader version.*may not be compatible.*", UserWarning)

        with OWAMcapReader(mcap_file_with_mixed_messages, decode_args=decode_args) as reader:
            messages = list(reader.iter_messages())
            assert len(messages) == 2

            for i, (msg, expected_type) in enumerate(zip(messages, expected_types)):
                if should_fail and expected_type is None:
                    # Should fail on unknown message type
                    with pytest.raises((ValueError, KeyError)):
                        _ = msg.decoded
                else:
                    # Should succeed
                    decoded = msg.decoded
                    assert isinstance(decoded, expected_type)

                    # Verify content is accessible
                    if isinstance(decoded, EasyDict):
                        # Dictionary access
                        if i == 0:  # Known message
                            assert decoded.data == "test"
                        else:  # Unknown message
                            assert decoded.value == 42
                    else:
                        # Typed access
                        assert decoded.data == "test"


def test_decode_args_type_validation():
    """Test that DecodeArgs type works correctly."""
    # Valid configurations
    valid_args: DecodeArgs = {"return_dict": True}
    assert valid_args["return_dict"] is True

    valid_args = {"return_dict_on_failure": True}
    assert valid_args["return_dict_on_failure"] is True

    valid_args = {"return_dict": False, "return_dict_on_failure": True}
    assert len(valid_args) == 2


def test_decode_args_edge_cases(mcap_file_with_mixed_messages):
    """Test edge cases and error conditions."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Domain-based message.*not found in registry.*", UserWarning)
        warnings.filterwarnings("ignore", "Reader version.*may not be compatible.*", UserWarning)

        # Empty decode_args (should use defaults) - will fail on unknown message types
        with OWAMcapReader(mcap_file_with_mixed_messages, decode_args={}) as reader:
            msg = next(reader.iter_messages())
            with pytest.raises(ValueError):  # Should fail for unknown message
                _ = msg.decoded

        # Redundant configuration (return_dict=True overrides return_dict_on_failure)
        with OWAMcapReader(
            mcap_file_with_mixed_messages, decode_args={"return_dict": True, "return_dict_on_failure": True}
        ) as reader:
            for msg in reader.iter_messages():
                decoded = msg.decoded
                assert isinstance(decoded, EasyDict)  # Always dict when return_dict=True


def test_decode_consistency(mcap_file_with_mixed_messages):
    """Test that decode results are consistent across multiple reads."""
    decode_configs = [
        {"return_dict": True},
        {"return_dict_on_failure": True},
    ]

    results = []
    for config in decode_configs:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Domain-based message.*not found in registry.*", UserWarning)
            warnings.filterwarnings("ignore", "Reader version.*may not be compatible.*", UserWarning)

            with OWAMcapReader(mcap_file_with_mixed_messages, decode_args=config) as reader:
                messages = []
                for msg in reader.iter_messages():
                    decoded = msg.decoded
                    # Convert to dict for comparison
                    if isinstance(decoded, OWAMessage):
                        content = decoded.model_dump()
                    else:
                        content = dict(decoded)
                    messages.append((msg.topic, msg.message_type, content))
                results.append(messages)

    # All configurations should produce the same content (different types, same data)
    assert len(results[0]) == len(results[1])
    for (topic1, type1, content1), (topic2, type2, content2) in zip(results[0], results[1]):
        assert topic1 == topic2
        assert type1 == type2
        assert content1 == content2


@pytest.fixture
def empty_mcap_file(tmp_path):
    """Create empty MCAP file for edge case testing."""
    file_path = tmp_path / "empty.mcap"
    with OWAMcapWriter(file_path) as writer:  # noqa: F841
        pass  # Write no messages
    return str(file_path)


def test_decode_args_with_empty_file(empty_mcap_file):
    """Test decode_args behavior with empty MCAP files."""
    for decode_args in [{}, {"return_dict": True}, {"return_dict_on_failure": True}]:
        with OWAMcapReader(empty_mcap_file, decode_args=decode_args) as reader:
            messages = list(reader.iter_messages())
            assert len(messages) == 0
            assert reader.message_count == 0
