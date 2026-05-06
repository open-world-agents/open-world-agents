"""
Tests for message base classes and verification (owa.core.message).
"""

import warnings

import pytest

from owa.core.message import OWAMessage


class ValidMessage(OWAMessage):
    """A message with a valid _type that uses domain-based format."""

    _type = "test/ValidMessage"  # Use domain-based format
    data: str


class TestMessageVerification:
    """Test cases for message type verification functionality."""

    def test_valid_message_verification(self):
        """Test that a message with valid _type passes verification."""
        # Domain-based messages should pass verification without warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ValidMessage.verify_type()
            assert result is True
            # No warnings should be issued for valid domain-based messages
            assert len(w) == 0

    def test_invalid_format_verification_old_style(self):
        """Test that a message with old module-based format fails verification."""

        class InvalidFormatMessage(OWAMessage):
            _type = "nonexistent.module.InvalidMessage"  # Old format should be rejected
            data: str

        with pytest.raises(ValueError, match="Invalid _type format.*Expected format: 'domain/MessageType'"):
            InvalidFormatMessage.verify_type()

    def test_invalid_format_verification(self):
        """Test that a message with invalid _type format fails verification."""

        class InvalidFormatMessage(OWAMessage):
            _type = "invalid_format"
            data: str

        with pytest.raises(ValueError, match="Invalid _type format 'invalid_format'"):
            InvalidFormatMessage.verify_type()

    def test_empty_type_verification(self):
        """Test that a message with empty _type fails verification."""

        class EmptyTypeMessage(OWAMessage):
            _type = ""
            data: str

        with pytest.raises(ValueError, match="must define a non-empty _type attribute"):
            EmptyTypeMessage.verify_type()

    def test_automatic_verification_on_creation(self):
        """Test that verification is automatically called when creating message instances."""
        # Valid domain-based message should create without warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            msg = ValidMessage(data="test")  # noqa: F841
            assert len(w) == 0  # No warnings for valid domain-based messages

    def test_old_format_rejection_on_creation(self):
        """Test that old module-based format is rejected during message creation."""

        class OldFormatMessage(OWAMessage):
            # This uses old module-based format which should be rejected
            _type = "owa.core.message.OWAMessage"
            data: str

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            msg = OldFormatMessage(data="test")  # noqa: F841
            # Should have a warning about verification failure
            assert len(w) == 1
            assert "Message type verification failed" in str(w[0].message)
            assert "Invalid _type format" in str(w[0].message)
