"""
Tests for message base classes and verification (owa.core.message).
"""

import warnings

import pytest

from owa.core.message import OWAMessage


class ValidMessage(OWAMessage):
    """A message with a valid _type that points to itself."""

    _type = "owa.core.message.OWAMessage"  # Use a valid existing class for testing
    data: str


class TestMessageVerification:
    """Test cases for message type verification functionality."""

    def test_valid_message_verification(self):
        """Test that a message with valid _type passes verification."""
        # This will issue a warning about class mismatch but still return True
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ValidMessage.verify_type()
            assert result is True
            # Should have a warning about class mismatch
            assert len(w) == 1
            assert "Class mismatch" in str(w[0].message)

    def test_invalid_module_verification(self):
        """Test that a message with invalid module in _type fails verification."""

        class InvalidModuleMessage(OWAMessage):
            _type = "nonexistent.module.InvalidMessage"
            data: str

        with pytest.raises(ImportError, match="Module 'nonexistent.module' specified in _type"):
            InvalidModuleMessage.verify_type()

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
        # Valid message should create with class mismatch warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            msg = ValidMessage(data="test")  # noqa: F841
            assert len(w) == 1  # Should have class mismatch warning
            assert "Class mismatch" in str(w[0].message)

    def test_class_mismatch_warning(self):
        """Test that a warning is issued when _type points to a different class."""

        class MismatchedMessage(OWAMessage):
            # This _type points to OWAMessage, not MismatchedMessage
            _type = "owa.core.message.OWAMessage"
            data: str

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = MismatchedMessage.verify_type()

            # Should still return True but issue a warning
            assert result is True
            assert len(w) == 1
            assert "Class mismatch" in str(w[0].message)
            assert "OWAMessage" in str(w[0].message)
            assert "MismatchedMessage" in str(w[0].message)
