"""
Tests for owa.core.lifecycle_node module.

This module tests the lifecycle node state machine functionality including
all state transitions and error handling.
"""

import pytest
from lifecycle_node import (
    InvalidStateTransitionError,
    LifecycleError,
    Node,
    NodeException,
    NodeStates,
    TransitionStates,
)


class TestNodeStates:
    """Test NodeStates enum."""

    def test_node_states_values(self):
        """Test that node states have correct string values."""
        assert NodeStates.UNCONFIGURED == "unconfigured"
        assert NodeStates.INACTIVE == "inactive"
        assert NodeStates.ACTIVE == "active"
        assert NodeStates.FINALIZED == "finalized"


class TestTransitionStates:
    """Test TransitionStates enum."""

    def test_transition_states_values(self):
        """Test that transition states have correct string values."""
        assert TransitionStates.CONFIGURING == "configuring"
        assert TransitionStates.CLEANING_UP == "cleaning_up"
        assert TransitionStates.SHUTTING_DOWN == "shutting_down"
        assert TransitionStates.ACTIVATING == "activating"
        assert TransitionStates.DEACTIVATING == "deactivating"
        assert TransitionStates.ERROR_PROCESSING == "error_processing"


class TestNodeExceptions:
    """Test node exception classes."""

    def test_node_exception_inheritance(self):
        """Test that node exceptions inherit correctly."""
        assert issubclass(InvalidStateTransitionError, NodeException)
        assert issubclass(LifecycleError, NodeException)
        assert issubclass(NodeException, Exception)

    def test_exception_creation(self):
        """Test that exceptions can be created with messages."""
        msg = "Test error message"

        node_exc = NodeException(msg)
        assert str(node_exc) == msg

        state_exc = InvalidStateTransitionError(msg)
        assert str(state_exc) == msg

        lifecycle_exc = LifecycleError(msg)
        assert str(lifecycle_exc) == msg


class TestNode:
    """Test Node lifecycle functionality."""

    def test_node_initialization(self):
        """Test that node initializes in UNCONFIGURED state."""
        node = Node()
        assert node.state == NodeStates.UNCONFIGURED

    def test_successful_lifecycle_flow(self):
        """Test complete successful lifecycle: configure -> activate -> deactivate -> cleanup."""
        node = Node()

        # Configure
        assert node.configure()
        assert node.state == NodeStates.INACTIVE

        # Activate
        assert node.activate()
        assert node.state == NodeStates.ACTIVE

        # Deactivate
        assert node.deactivate()
        assert node.state == NodeStates.INACTIVE

        # Cleanup
        assert node.cleanup()
        assert node.state == NodeStates.UNCONFIGURED

    def test_shutdown_from_various_states(self):
        """Test shutdown can be called from UNCONFIGURED, INACTIVE, and ACTIVE states."""
        # From UNCONFIGURED
        node = Node()
        assert node.shutdown()
        assert node.state == NodeStates.FINALIZED

        # From INACTIVE
        node = Node()
        node.configure()
        assert node.shutdown()
        assert node.state == NodeStates.FINALIZED

        # From ACTIVE
        node = Node()
        node.configure()
        node.activate()
        assert node.shutdown()
        assert node.state == NodeStates.FINALIZED

    def test_destroy_from_finalized(self):
        """Test destroy can only be called from FINALIZED state."""
        node = Node()
        node.shutdown()
        assert node.state == NodeStates.FINALIZED

        node.destroy()
        assert node.state is None

    def test_invalid_state_transitions(self):
        """Test that invalid state transitions raise exceptions."""
        node = Node()

        # Cannot activate from UNCONFIGURED
        with pytest.raises(InvalidStateTransitionError, match="Can only activate from Inactive state"):
            node.activate()

        # Cannot deactivate from UNCONFIGURED
        with pytest.raises(InvalidStateTransitionError, match="Can only deactivate from Active state"):
            node.deactivate()

        # Configure first
        node.configure()

        # Cannot configure again from INACTIVE
        with pytest.raises(InvalidStateTransitionError, match="Can only configure from Unconfigured state"):
            node.configure()

        # Cannot deactivate from INACTIVE
        with pytest.raises(InvalidStateTransitionError, match="Can only deactivate from Active state"):
            node.deactivate()

        # Activate
        node.activate()

        # Cannot configure from ACTIVE
        with pytest.raises(InvalidStateTransitionError, match="Can only configure from Unconfigured state"):
            node.configure()

        # Cannot activate again from ACTIVE
        with pytest.raises(InvalidStateTransitionError, match="Can only activate from Inactive state"):
            node.activate()

    def test_destroy_invalid_state(self):
        """Test that destroy raises exception from non-FINALIZED states."""
        node = Node()

        with pytest.raises(InvalidStateTransitionError, match="Can only destroy from Finalized state"):
            node.destroy()

        node.configure()
        with pytest.raises(InvalidStateTransitionError, match="Can only destroy from Finalized state"):
            node.destroy()

    def test_cleanup_invalid_state(self):
        """Test that cleanup raises exception from invalid states."""
        node = Node()

        # Cannot cleanup from UNCONFIGURED
        with pytest.raises(
            InvalidStateTransitionError, match="Can only clean up from Inactive or Error Processing state"
        ):
            node.cleanup()

        node.configure()
        node.activate()

        # Cannot cleanup from ACTIVE
        with pytest.raises(
            InvalidStateTransitionError, match="Can only clean up from Inactive or Error Processing state"
        ):
            node.cleanup()

    def test_shutdown_invalid_state(self):
        """Test that shutdown raises exception from invalid states."""
        node = Node()
        node.shutdown()  # Move to FINALIZED

        # Cannot shutdown from FINALIZED
        with pytest.raises(InvalidStateTransitionError, match="Cannot shutdown from current state"):
            node.shutdown()


class FailingNode(Node):
    """Test node that fails lifecycle methods."""

    def __init__(
        self,
        fail_configure=False,
        fail_activate=False,
        fail_deactivate=False,
        fail_cleanup=False,
        fail_shutdown=False,
        fail_error=False,
    ):
        super().__init__()
        self.fail_configure = fail_configure
        self.fail_activate = fail_activate
        self.fail_deactivate = fail_deactivate
        self.fail_cleanup = fail_cleanup
        self.fail_shutdown = fail_shutdown
        self.fail_error = fail_error

    def on_configure(self):
        if self.fail_configure:
            return False
        return True

    def on_activate(self):
        if self.fail_activate:
            return False
        return True

    def on_deactivate(self):
        if self.fail_deactivate:
            return False
        return True

    def on_cleanup(self):
        if self.fail_cleanup:
            return False
        return True

    def on_shutdown(self):
        if self.fail_shutdown:
            return False
        return True

    def on_error(self):
        if self.fail_error:
            return False
        return True


class TestNodeErrorHandling:
    """Test node error handling and recovery."""

    def test_configure_failure(self):
        """Test configure failure triggers error handling."""
        node = FailingNode(fail_configure=True)

        assert not node.configure()
        assert node.state == NodeStates.UNCONFIGURED  # Error handling should reset to UNCONFIGURED

    def test_activate_failure(self):
        """Test activate failure triggers error handling."""
        node = FailingNode(fail_activate=True)
        node.configure()

        assert not node.activate()
        assert node.state == NodeStates.UNCONFIGURED  # Error handling should reset to UNCONFIGURED

    def test_deactivate_failure(self):
        """Test deactivate failure triggers error handling."""
        node = FailingNode(fail_deactivate=True)
        node.configure()
        node.activate()

        assert not node.deactivate()
        assert node.state == NodeStates.UNCONFIGURED  # Error handling should reset to UNCONFIGURED

    def test_cleanup_failure(self):
        """Test cleanup failure triggers error handling."""
        node = FailingNode(fail_cleanup=True)
        node.configure()
        node.activate()
        node.deactivate()

        assert not node.cleanup()
        assert node.state == NodeStates.UNCONFIGURED  # Error handling should reset to UNCONFIGURED

    def test_shutdown_failure(self):
        """Test shutdown failure triggers error handling."""
        node = FailingNode(fail_shutdown=True)

        assert not node.shutdown()
        assert node.state == NodeStates.UNCONFIGURED  # Error handling should reset to UNCONFIGURED

    def test_error_handler_failure(self):
        """Test error handler failure leads to FINALIZED state."""
        node = FailingNode(fail_configure=True, fail_error=True)

        assert not node.configure()
        assert node.state == NodeStates.FINALIZED  # Failed error handling should lead to FINALIZED

    def test_cleanup_from_error_processing(self):
        """Test cleanup can be called from ERROR_PROCESSING state."""
        node = FailingNode(fail_configure=True)
        node.configure()  # This will fail and put node in ERROR_PROCESSING temporarily

        # Manually set to ERROR_PROCESSING to test cleanup from this state
        node.state = TransitionStates.ERROR_PROCESSING
        assert node.cleanup()
        assert node.state == NodeStates.UNCONFIGURED


class ExceptionThrowingNode(Node):
    """Test node that throws exceptions in lifecycle methods."""

    def __init__(self, throw_in_configure=False, throw_in_activate=False, throw_in_error=False):
        super().__init__()
        self.throw_in_configure = throw_in_configure
        self.throw_in_activate = throw_in_activate
        self.throw_in_error = throw_in_error

    def on_configure(self):
        if self.throw_in_configure:
            raise RuntimeError("Configure failed with exception")
        return True

    def on_activate(self):
        if self.throw_in_activate:
            raise RuntimeError("Activate failed with exception")
        return True

    def on_error(self):
        if self.throw_in_error:
            raise RuntimeError("Error handler failed with exception")
        return True


class TestNodeExceptionHandling:
    """Test node exception handling in lifecycle methods."""

    def test_configure_exception_handling(self, capsys):
        """Test that exceptions in configure are caught and handled."""
        node = ExceptionThrowingNode(throw_in_configure=True)

        assert not node.configure()
        assert node.state == NodeStates.UNCONFIGURED

        # Check that error was printed
        captured = capsys.readouterr()
        assert "Error in on_configure: Configure failed with exception" in captured.out

    def test_activate_exception_handling(self, capsys):
        """Test that exceptions in activate are caught and handled."""
        node = ExceptionThrowingNode(throw_in_activate=True)
        node.configure()

        assert not node.activate()
        assert node.state == NodeStates.UNCONFIGURED

        # Check that error was printed
        captured = capsys.readouterr()
        assert "Error in on_activate: Activate failed with exception" in captured.out

    def test_error_handler_exception_handling(self, capsys):
        """Test that exceptions in error handler are caught."""
        node = ExceptionThrowingNode(throw_in_configure=True, throw_in_error=True)

        assert not node.configure()
        assert node.state == NodeStates.FINALIZED

        # Check that both errors were printed
        captured = capsys.readouterr()
        assert "Error in on_configure: Configure failed with exception" in captured.out
        assert "Error in on_error: Error handler failed with exception" in captured.out


class TestNodeBaseMethods:
    """Test base Node class methods."""

    def test_base_on_error_method(self):
        """Test that base on_error method returns True."""
        node = Node()

        # Test the base on_error method directly
        result = node.on_error()
        assert result is True
