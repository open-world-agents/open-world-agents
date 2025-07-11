"""
Tests for owa.core.listener module.

This module tests the listener functionality including listener lifecycle,
configuration, and error handling.
"""

import inspect
import threading
import time
from unittest.mock import patch

import pytest

from owa.core.listener import Listener, ListenerMixin, ListenerProcess, ListenerThread


class TestListenerMixin:
    """Test ListenerMixin functionality."""

    def test_callback_registration_and_access(self):
        """Test callback registration and property access."""

        class TestListener(ListenerMixin):
            def loop(self):
                pass

            # Implement abstract methods from RunnableMixin
            def is_alive(self):
                return False

            def start(self):
                pass

            def stop(self):
                pass

            def join(self, timeout=None):
                pass

        listener = TestListener()

        # Test initial state - callback should raise AttributeError when not set
        with pytest.raises(AttributeError, match="Callback not set"):
            listener.get_callback()

        with pytest.raises(AttributeError, match="Callback not set"):
            _ = listener.callback

        # Test callback registration
        def test_callback():
            pass

        listener.register_callback(test_callback)
        assert listener.get_callback() is test_callback
        assert listener.callback is test_callback

        # Test callback property setter
        def another_callback():
            pass

        listener.callback = another_callback
        assert listener.callback is another_callback

    def test_configure_with_callback(self):
        """Test configure method with callback parameter."""

        class TestListener(ListenerMixin):
            def loop(self):
                pass

            # Implement abstract methods from RunnableMixin
            def is_alive(self):
                return False

            def start(self):
                pass

            def stop(self):
                pass

            def join(self, timeout=None):
                pass

            def configure(self, *args, **kwargs):
                # Mock parent configure method
                self._configured = True
                return super().configure(*args, **kwargs)

        listener = TestListener()

        def test_callback():
            pass

        # Test configure with callback
        result = listener.configure(callback=test_callback, some_param="value")

        assert result is listener  # Should return self
        assert listener.callback is test_callback
        assert hasattr(listener, "_configured")

    def test_configure_without_callback_raises_error(self):
        """Test that configure raises TypeError when callback is missing."""

        class TestListener(ListenerMixin):
            def loop(self):
                pass

            # Implement abstract methods from RunnableMixin
            def is_alive(self):
                return False

            def start(self):
                pass

            def stop(self):
                pass

            def join(self, timeout=None):
                pass

        listener = TestListener()

        # Should raise TypeError for missing required callback argument
        with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'callback'"):
            listener.configure()


class ConcreteListenerThread(ListenerThread):
    """Concrete implementation of ListenerThread for testing."""

    def __init__(self):
        super().__init__()
        self.loop_called = False
        self.loop_args = None
        self.loop_kwargs = None

    def loop(self, stop_event=None, callback=None):
        """Test implementation of loop method."""
        self.loop_called = True
        self.loop_args = (stop_event,)
        self.loop_kwargs = {"callback": callback}

        # Simulate some work and respect stop_event
        if stop_event:
            while not stop_event.is_set():
                time.sleep(0.01)
                if callback:
                    callback("test_data")
                break  # Exit after one iteration for testing


class TestListenerThread:
    """Test ListenerThread functionality."""

    def test_run_method_checks_configuration(self):
        """Test that run method checks if listener is configured."""
        listener = ConcreteListenerThread()

        # Should raise RuntimeError if not configured
        with pytest.raises(RuntimeError, match="RunnableThread is not configured"):
            listener.run()

    def test_run_method_with_configuration(self):
        """Test run method with proper configuration."""
        listener = ConcreteListenerThread()

        def test_callback(data):
            pass

        # Configure the listener
        listener.configure(callback=test_callback)

        # Run should work now
        listener.run()

        assert listener.loop_called
        assert isinstance(listener.loop_args[0], threading.Event)
        assert listener.loop_kwargs["callback"] is test_callback

    def test_run_method_parameter_inspection(self):
        """Test that run method inspects loop signature and passes appropriate parameters."""

        class MinimalListener(ListenerThread):
            def __init__(self):
                super().__init__()
                self.loop_called = False
                self.received_params = {}

            def loop(self):
                """Loop without parameters."""
                self.loop_called = True

        listener = MinimalListener()
        listener.configure(callback=lambda: None)

        listener.run()
        assert listener.loop_called

    def test_run_method_with_stop_event_parameter(self):
        """Test run method when loop accepts stop_event parameter."""

        class StopEventListener(ListenerThread):
            def __init__(self):
                super().__init__()
                self.received_stop_event = None

            def loop(self, stop_event):
                self.received_stop_event = stop_event

        listener = StopEventListener()
        listener.configure(callback=lambda: None)

        listener.run()
        assert isinstance(listener.received_stop_event, threading.Event)

    def test_run_method_with_callback_parameter(self):
        """Test run method when loop accepts callback parameter."""

        class CallbackListener(ListenerThread):
            def __init__(self):
                super().__init__()
                self.received_callback = None

            def loop(self, callback):
                self.received_callback = callback

        def test_callback():
            pass

        listener = CallbackListener()
        listener.configure(callback=test_callback)

        listener.run()
        assert listener.received_callback is test_callback

    def test_full_lifecycle(self):
        """Test complete listener lifecycle: configure, start, stop, join."""
        listener = ConcreteListenerThread()
        events_received = []

        def callback(data):
            events_received.append(data)

        # Configure
        listener.configure(callback=callback)

        # Start
        listener.start()
        assert listener.is_alive()

        # Let it run briefly
        time.sleep(0.05)

        # Stop
        listener.stop()
        listener.join(timeout=1.0)

        assert not listener.is_alive()
        assert len(events_received) > 0
        assert "test_data" in events_received


class ConcreteListenerProcess(ListenerProcess):
    """Concrete implementation of ListenerProcess for testing."""

    def loop(self, stop_event=None, callback=None):
        """Test implementation of loop method."""
        # Simulate some work and respect stop_event
        if stop_event:
            while not stop_event.is_set():
                time.sleep(0.01)
                break  # Exit after one iteration for testing


class TestListenerProcess:
    """Test ListenerProcess functionality."""

    def test_run_method_checks_configuration(self):
        """Test that run method checks if listener is configured."""
        listener = ConcreteListenerProcess()

        # Should raise RuntimeError if not configured
        with pytest.raises(RuntimeError, match="RunnableProcess is not configured"):
            listener.run()

    def test_run_method_with_configuration(self):
        """Test run method with proper configuration."""
        listener = ConcreteListenerProcess()

        def test_callback(data):
            pass

        # Configure the listener
        listener.configure(callback=test_callback)

        # Mock the loop method to avoid actual process execution in tests
        with patch.object(listener, "loop") as mock_loop:
            listener.run()
            mock_loop.assert_called_once()

    def test_run_method_parameter_inspection(self):
        """Test that run method inspects loop signature for process."""

        class MinimalProcessListener(ListenerProcess):
            def loop(self):
                """Loop without parameters."""
                pass

        listener = MinimalProcessListener()
        listener.configure(callback=lambda: None)

        with patch.object(listener, "loop") as mock_loop:
            listener.run()
            mock_loop.assert_called_once_with()

    def test_run_method_with_stop_event_parameter(self):
        """Test run method when process loop accepts stop_event parameter."""

        class StopEventProcessListener(ListenerProcess):
            def loop(self, stop_event):
                pass

        listener = StopEventProcessListener()
        listener.configure(callback=lambda: None)

        # Mock inspect.signature to ensure stop_event parameter is detected
        original_signature = inspect.signature

        def mock_signature(func):
            if func == listener.loop:
                return inspect.Signature(
                    [
                        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                        inspect.Parameter("stop_event", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                    ]
                )
            return original_signature(func)

        with patch("inspect.signature", side_effect=mock_signature):
            with patch.object(listener, "loop") as mock_loop:
                listener.run()
                mock_loop.assert_called_once()
                # Check that stop_event was passed
                args, kwargs = mock_loop.call_args
                assert "stop_event" in kwargs
                # Check that it's a multiprocessing Event-like object
                stop_event = kwargs["stop_event"]
                assert hasattr(stop_event, "is_set")
                assert hasattr(stop_event, "set")


class TestListenerAlias:
    """Test that Listener is properly aliased to ListenerThread."""

    def test_listener_is_listener_thread(self):
        """Test that Listener is an alias for ListenerThread."""
        assert Listener is ListenerThread

    def test_listener_instantiation(self):
        """Test that Listener can be instantiated as ListenerThread."""

        class TestListener(Listener):
            def loop(self, stop_event, callback):
                pass

        listener = TestListener()
        assert isinstance(listener, ListenerThread)
        assert isinstance(listener, ListenerMixin)


class TestListenerErrorHandling:
    """Test error handling in listener implementations."""

    def test_callback_error_handling(self):
        """Test that errors in callbacks don't crash the listener."""

        class ErrorHandlingListener(ListenerThread):
            def __init__(self):
                super().__init__()
                self.error_occurred = False

            def loop(self, stop_event, callback):
                try:
                    callback("test_data")
                except Exception:
                    self.error_occurred = True
                    # Continue running despite callback error

        def failing_callback(data):
            raise ValueError("Callback failed")

        listener = ErrorHandlingListener()
        listener.configure(callback=failing_callback)

        listener.run()
        assert listener.error_occurred

    def test_unconfigured_listener_error_message(self):
        """Test specific error message for unconfigured listeners."""
        listener = ConcreteListenerThread()

        with pytest.raises(RuntimeError) as exc_info:
            listener.run()

        error_msg = str(exc_info.value)
        assert "RunnableThread is not configured" in error_msg
        assert "Call configure() before start()" in error_msg
        assert "overriden the configure method" in error_msg


class TestListenerInspection:
    """Test parameter inspection functionality."""

    def test_inspect_signature_with_various_parameters(self):
        """Test that parameter inspection works with different method signatures."""

        class VariousParamListener(ListenerThread):
            def __init__(self):
                super().__init__()
                self.received_params = {}

            def loop(self, stop_event, callback, extra_param=None):
                self.received_params = {"stop_event": stop_event, "callback": callback, "extra_param": extra_param}

        listener = VariousParamListener()
        listener.configure(callback=lambda: None)

        # Mock inspect.signature to test parameter detection
        original_signature = inspect.signature

        def mock_signature(func):
            if func == listener.loop:
                # Return a signature that includes stop_event and callback
                import inspect

                return inspect.Signature(
                    [
                        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                        inspect.Parameter("stop_event", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                        inspect.Parameter("callback", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                    ]
                )
            return original_signature(func)

        with patch("inspect.signature", side_effect=mock_signature):
            listener.run()

        assert "stop_event" in listener.received_params
        assert "callback" in listener.received_params
