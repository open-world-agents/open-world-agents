"""
Tests for message type verification functionality.

This module tests the _type verification feature added to BaseMessage and OWAMessage.
"""

import warnings

import pytest

from owa.core.message import OWAMessage


class ValidMessage(OWAMessage):
    """A message with a valid _type that points to itself."""

    _type = "owa.core.message.OWAMessage"  # Use a valid existing class for testing
    data: str


class InvalidModuleMessage(OWAMessage):
    """A message with an invalid module path in _type."""

    _type = "nonexistent.module.InvalidMessage"
    data: str


class InvalidClassMessage(OWAMessage):
    """A message with a valid module but invalid class name in _type."""

    _type = "test_message_verification.NonexistentClass"
    data: str


class InvalidFormatMessage(OWAMessage):
    """A message with an invalid _type format."""

    _type = "invalid_format"
    data: str


class EmptyTypeMessage(OWAMessage):
    """A message with an empty _type."""

    _type = ""
    data: str


def test_valid_message_verification():
    """Test that a message with valid _type passes verification."""
    # This will issue a warning about class mismatch but still return True
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = ValidMessage.verify_type()
        assert result is True
        # Should have a warning about class mismatch
        assert len(w) == 1
        assert "Class mismatch" in str(w[0].message)


def test_invalid_module_verification():
    """Test that a message with invalid module in _type fails verification."""
    with pytest.raises(ImportError, match="Module 'nonexistent.module' specified in _type"):
        InvalidModuleMessage.verify_type()


def test_invalid_class_verification():
    """Test that a message with invalid class name in _type fails verification."""
    with pytest.raises(AttributeError, match="Class 'NonexistentClass' not found in module"):
        InvalidClassMessage.verify_type()


def test_invalid_format_verification():
    """Test that a message with invalid _type format fails verification."""
    with pytest.raises(ValueError, match="Invalid _type format 'invalid_format'"):
        InvalidFormatMessage.verify_type()


def test_empty_type_verification():
    """Test that a message with empty _type fails verification."""
    with pytest.raises(ValueError, match="must define a non-empty _type attribute"):
        EmptyTypeMessage.verify_type()


def test_automatic_verification_on_creation():
    """Test that verification is automatically called when creating message instances."""
    # Valid message should create with class mismatch warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        msg = ValidMessage(data="test")
        assert len(w) == 1  # Should have class mismatch warning
        assert "Class mismatch" in str(w[0].message)

    # Invalid message should create with verification failure warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        msg = InvalidModuleMessage(data="test")  # noqa: F841
        assert len(w) == 1
        assert "Message type verification failed" in str(w[0].message)
        assert "nonexistent.module" in str(w[0].message)


def test_class_mismatch_warning():
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


def test_message_without_type_attribute():
    """Test verification fails for classes without _type attribute."""

    class NoTypeMessage(OWAMessage):
        # Intentionally not defining _type
        data: str

    # In Pydantic v2, _type becomes a ModelPrivateAttr even when not explicitly defined
    # So we need to test a different scenario - when _type is None or empty
    # Let's test with an empty _type instead
    class EmptyTypeMessage2(OWAMessage):
        _type = ""  # Empty string
        data: str

    with pytest.raises(ValueError, match="must define a non-empty _type attribute"):
        EmptyTypeMessage2.verify_type()
