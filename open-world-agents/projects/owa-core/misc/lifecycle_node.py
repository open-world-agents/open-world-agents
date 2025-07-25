"""
This is not being used currently!
"""

# References: https://foxglove.dev/blog/how-to-use-ros2-lifecycle-nodes

from enum import StrEnum


class NodeStates(StrEnum):
    UNCONFIGURED = "unconfigured"
    INACTIVE = "inactive"
    ACTIVE = "active"
    FINALIZED = "finalized"


class TransitionStates(StrEnum):
    CONFIGURING = "configuring"
    CLEANING_UP = "cleaning_up"
    SHUTTING_DOWN = "shutting_down"
    ACTIVATING = "activating"
    DEACTIVATING = "deactivating"
    ERROR_PROCESSING = "error_processing"


class NodeException(Exception):
    """Base exception class for Node errors."""

    pass


class InvalidStateTransitionError(NodeException):
    """Raised when an invalid state transition is attempted."""

    pass


class LifecycleError(NodeException):
    """Raised when a lifecycle method fails."""

    pass


class Node:
    def __init__(self):
        """The constructor of the Node class. This must be called by the subclass."""
        self.state = NodeStates.UNCONFIGURED

    def configure(self, *args, **kwargs):
        if self.state != NodeStates.UNCONFIGURED:
            raise InvalidStateTransitionError("Can only configure from Unconfigured state.")
        self.state = TransitionStates.CONFIGURING
        try:
            if not self.on_configure(*args, **kwargs):
                raise LifecycleError("Configuration failed.")
        except Exception as e:
            print(f"Error in on_configure: {e}")
            self.handle_error()
            return False
        self.state = NodeStates.INACTIVE
        return True

    def activate(self, *args, **kwargs):
        if self.state != NodeStates.INACTIVE:
            raise InvalidStateTransitionError("Can only activate from Inactive state.")
        self.state = TransitionStates.ACTIVATING
        try:
            if not self.on_activate(*args, **kwargs):
                raise LifecycleError("Activation failed.")
        except Exception as e:
            print(f"Error in on_activate: {e}")
            self.handle_error()
            return False
        self.state = NodeStates.ACTIVE
        return True

    def deactivate(self, *args, **kwargs):
        if self.state != NodeStates.ACTIVE:
            raise InvalidStateTransitionError("Can only deactivate from Active state.")
        self.state = TransitionStates.DEACTIVATING
        try:
            if not self.on_deactivate(*args, **kwargs):
                raise LifecycleError("Deactivation failed.")
        except Exception as e:
            print(f"Error in on_deactivate: {e}")
            self.handle_error()
            return False
        self.state = NodeStates.INACTIVE
        return True

    def cleanup(self, *args, **kwargs):
        if self.state not in [NodeStates.INACTIVE, TransitionStates.ERROR_PROCESSING]:
            raise InvalidStateTransitionError("Can only clean up from Inactive or Error Processing state.")
        self.state = TransitionStates.CLEANING_UP
        try:
            if not self.on_cleanup(*args, **kwargs):
                raise LifecycleError("Cleanup failed.")
        except Exception as e:
            print(f"Error in on_cleanup: {e}")
            self.handle_error()
            return False
        self.state = NodeStates.UNCONFIGURED
        return True

    def shutdown(self, *args, **kwargs):
        if self.state in [NodeStates.UNCONFIGURED, NodeStates.INACTIVE, NodeStates.ACTIVE]:
            self.state = TransitionStates.SHUTTING_DOWN
            try:
                if not self.on_shutdown(*args, **kwargs):
                    raise LifecycleError("Shutdown failed.")
            except Exception as e:
                print(f"Error in on_shutdown: {e}")
                self.handle_error()
                return False
            self.state = NodeStates.FINALIZED
            return True
        else:
            raise InvalidStateTransitionError("Cannot shutdown from current state.")

    def destroy(self):
        if self.state != NodeStates.FINALIZED:
            raise InvalidStateTransitionError("Can only destroy from Finalized state.")
        self.state = None

    def handle_error(self):
        self.state = TransitionStates.ERROR_PROCESSING
        try:
            if self.on_error():
                self.state = NodeStates.UNCONFIGURED
            else:
                self.state = NodeStates.FINALIZED
        except Exception as e:
            print(f"Error in on_error: {e}")
            self.state = NodeStates.FINALIZED

    # Override these methods in a subclass
    def on_configure(self):
        return True

    def on_activate(self):
        return True

    def on_deactivate(self):
        return True

    def on_cleanup(self):
        return True

    def on_shutdown(self):
        return True

    def on_error(self):
        return True
