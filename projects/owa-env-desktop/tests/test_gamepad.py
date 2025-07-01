"""
Tests for gamepad listeners and callables.

This module tests the gamepad functionality including listeners and callables.
"""

import pytest
from unittest.mock import patch, MagicMock

from owa.env.desktop.gamepad.listeners import GamepadListenerWrapper, GamepadStateListener
from owa.env.desktop.gamepad.callables import (
    get_gamepad_state, 
    get_connected_gamepads, 
    is_gamepad_connected,
    get_gamepad_events
)


class TestGamepadCallables:
    """Test gamepad callable functions."""

    def test_get_gamepad_state_no_inputs_library(self):
        """Test get_gamepad_state when inputs library is not available."""
        with patch('owa.env.desktop.gamepad.callables.INPUTS_AVAILABLE', False):
            with pytest.raises(ImportError, match="inputs.*library is required"):
                get_gamepad_state()

    def test_get_connected_gamepads_no_inputs_library(self):
        """Test get_connected_gamepads when inputs library is not available."""
        with patch('owa.env.desktop.gamepad.callables.INPUTS_AVAILABLE', False):
            with pytest.raises(ImportError, match="inputs.*library is required"):
                get_connected_gamepads()

    def test_is_gamepad_connected_no_inputs_library(self):
        """Test is_gamepad_connected when inputs library is not available."""
        with patch('owa.env.desktop.gamepad.callables.INPUTS_AVAILABLE', False):
            with pytest.raises(ImportError, match="inputs.*library is required"):
                is_gamepad_connected()

    def test_get_gamepad_events_no_inputs_library(self):
        """Test get_gamepad_events when inputs library is not available."""
        with patch('owa.env.desktop.gamepad.callables.INPUTS_AVAILABLE', False):
            with pytest.raises(ImportError, match="inputs.*library is required"):
                get_gamepad_events()

    @patch('owa.env.desktop.gamepad.callables.INPUTS_AVAILABLE', True)
    @patch('owa.env.desktop.gamepad.callables.inputs')
    def test_get_gamepad_state_no_gamepads(self, mock_inputs):
        """Test get_gamepad_state when no gamepads are connected."""
        # Mock DeviceManager with no gamepads
        mock_device_manager = MagicMock()
        mock_device_manager.gamepads = []
        mock_inputs.DeviceManager.return_value = mock_device_manager
        
        result = get_gamepad_state(0)
        assert result is None

    @patch('owa.env.desktop.gamepad.callables.INPUTS_AVAILABLE', True)
    @patch('owa.env.desktop.gamepad.callables.inputs')
    def test_get_connected_gamepads_empty(self, mock_inputs):
        """Test get_connected_gamepads when no gamepads are connected."""
        # Mock DeviceManager with no gamepads
        mock_device_manager = MagicMock()
        mock_device_manager.gamepads = []
        mock_inputs.DeviceManager.return_value = mock_device_manager
        
        result = get_connected_gamepads()
        assert result == []

    @patch('owa.env.desktop.gamepad.callables.INPUTS_AVAILABLE', True)
    @patch('owa.env.desktop.gamepad.callables.inputs')
    def test_is_gamepad_connected_false(self, mock_inputs):
        """Test is_gamepad_connected when no gamepads are connected."""
        # Mock DeviceManager with no gamepads
        mock_device_manager = MagicMock()
        mock_device_manager.gamepads = []
        mock_inputs.DeviceManager.return_value = mock_device_manager
        
        result = is_gamepad_connected(0)
        assert result is False


class TestGamepadListeners:
    """Test gamepad listener classes."""

    def test_gamepad_listener_wrapper_creation(self):
        """Test that GamepadListenerWrapper can be created."""
        listener = GamepadListenerWrapper()
        assert listener is not None

    def test_gamepad_state_listener_creation(self):
        """Test that GamepadStateListener can be created."""
        listener = GamepadStateListener()
        assert listener is not None

    @patch('owa.env.desktop.gamepad.listeners.get_gamepad_events')
    def test_gamepad_listener_wrapper_loop_no_events(self, mock_get_events):
        """Test GamepadListenerWrapper loop when no events are available."""
        mock_get_events.return_value = []
        
        listener = GamepadListenerWrapper()
        callback = MagicMock()
        listener.configure(callback=callback)
        
        # Mock stop_event that's immediately set
        stop_event = MagicMock()
        stop_event.is_set.side_effect = [False, True]  # Run once then stop
        
        with patch('time.sleep') as mock_sleep:
            listener.loop(stop_event)
            
        # Should have called sleep since no events
        mock_sleep.assert_called_once_with(0.01)
        callback.assert_not_called()

    @patch('owa.env.desktop.gamepad.listeners.get_gamepad_state')
    def test_gamepad_state_listener_loop_no_gamepad(self, mock_get_state):
        """Test GamepadStateListener loop when no gamepad is connected."""
        mock_get_state.return_value = None
        
        listener = GamepadStateListener()
        callback = MagicMock()
        listener.configure(callback=callback)
        
        # Mock stop_event that's immediately set
        stop_event = MagicMock()
        stop_event.is_set.side_effect = [False, True]  # Run once then stop
        stop_event.wait = MagicMock()
        
        listener.loop(stop_event)
        
        # Should not have called callback since no gamepad
        callback.assert_not_called()
        stop_event.wait.assert_called_once_with(1.0)

    @patch('owa.env.desktop.gamepad.listeners.get_gamepad_state')
    def test_gamepad_state_listener_loop_with_gamepad(self, mock_get_state):
        """Test GamepadStateListener loop when gamepad is connected."""
        from owa.msgs.desktop.gamepad import GamepadState
        
        # Mock a gamepad state
        mock_state = GamepadState(
            gamepad_type="GAMEPAD_TYPE_STANDARD",
            buttons=set(),
            axes={},
            timestamp=123456789
        )
        mock_get_state.return_value = mock_state
        
        listener = GamepadStateListener()
        callback = MagicMock()
        listener.configure(callback=callback)
        
        # Mock stop_event that's immediately set
        stop_event = MagicMock()
        stop_event.is_set.side_effect = [False, True]  # Run once then stop
        stop_event.wait = MagicMock()
        
        listener.loop(stop_event)
        
        # Should have called callback with the state
        callback.assert_called_once_with(mock_state)
        stop_event.wait.assert_called_once_with(1.0)
