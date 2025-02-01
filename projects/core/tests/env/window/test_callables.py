# tests/env/window/test_callables.py

import platform
import sys
from unittest import mock

import pytest

# Import the module and functions you want to test
from owa.env.window import callables
from owa.env.window.msg import WindowInfo

# Skip tests if the OS is not supported
supported_os = ["Windows", "Darwin"]  # Add 'Linux' if applicable


@pytest.fixture
def mock_windows_active_window():
    with mock.patch("pygetwindow.getActiveWindow") as mock_active_window:
        mock_window = mock.Mock()
        mock_window._hWnd = 123456
        mock_window.title = "Test Window"
        rect = mock.Mock()
        rect.left = 100
        rect.top = 100
        rect.right = 500
        rect.bottom = 400
        mock_window._getWindowRect.return_value = rect
        mock_active_window.return_value = mock_window
        yield


@pytest.fixture
def mock_macos_active_window():
    with mock.patch("Quartz.CGWindowListCopyWindowInfo") as mock_cgwindow_list:
        mock_window_info = [
            {
                "kCGWindowLayer": 0,
                "kCGWindowName": "Test Window",
                "kCGWindowBounds": {"X": 100, "Y": 100, "Width": 400, "Height": 300},
                "kCGWindowNumber": 123456,
            }
        ]
        mock_cgwindow_list.return_value = mock_window_info
        yield


@pytest.mark.skipif(platform.system() not in supported_os, reason="OS not supported")
def test_get_active_window_windows(mock_windows_active_window):
    if platform.system() == "Windows":
        result = callables.get_active_window()
        assert result is not None
        assert isinstance(result, dict)
        assert result["title"] == "Test Window"
        assert result["rect"] == (100, 100, 500, 400)
        assert result["hWnd"] == 123456


@pytest.mark.skipif(platform.system() not in supported_os, reason="OS not supported")
def test_get_active_window_macos(mock_macos_active_window):
    if platform.system() == "Darwin":
        result = callables.get_active_window()
        assert result is not None
        assert isinstance(result, dict)
        assert result["title"] == "Test Window"
        assert result["rect"] == (100, 100, 500, 400)
        assert result["hWnd"] == 123456


def test_get_window_by_title_windows(monkeypatch):
    if platform.system() == "Windows":
        from pygetwindow import Window

        mock_window = Window(title="Test Window")
        mock_window._hWnd = 123456
        rect = mock.Mock()
        rect.left = 100
        rect.top = 100
        rect.right = 500
        rect.bottom = 400
        mock_window._getWindowRect.return_value = rect

        with mock.patch("pygetwindow.getWindowsWithTitle", return_value=[mock_window]):
            result = callables.get_window_by_title("Test")
            assert result is not None
            assert isinstance(result, WindowInfo)
            assert result.title == "Test Window"
            assert result.rect == (100, 100, 500, 400)
            assert result.hWnd == 123456


def test_get_window_by_title_macos(monkeypatch):
    if platform.system() == "Darwin":
        mock_window_info = [
            {
                "kCGWindowLayer": 0,
                "kCGWindowName": "Test Window",
                "kCGWindowBounds": {"X": 100, "Y": 100, "Width": 400, "Height": 300},
                "kCGWindowNumber": 123456,
            }
        ]
        with mock.patch("Quartz.CGWindowListCopyWindowInfo", return_value=mock_window_info):
            result = callables.get_window_by_title("Test")
            assert result is not None
            assert isinstance(result, WindowInfo)
            assert result.title == "Test Window"
            assert result.rect == (100, 100, 500, 400)
            assert result.hWnd == 123456


def test_when_active_decorator_windows(monkeypatch):
    if platform.system() == "Windows":
        # Mock get_active_window and get_window_by_title
        with mock.patch("owa.env.window.callables.get_window_by_title") as mock_get_window_by_title:
            mock_window_info = WindowInfo(title="Test Window", rect=(100, 100, 500, 400), hWnd=123456)
            mock_get_window_by_title.return_value = mock_window_info

            with mock.patch("pygetwindow.getActiveWindow") as mock_get_active_window:
                mock_active_window = mock.Mock()
                mock_active_window._hWnd = 123456
                mock_get_active_window.return_value = mock_active_window

                @callables.when_active("Test")
                def sample_function():
                    return "Function Executed"

                result = sample_function()
                assert result == "Function Executed"


def test_when_active_decorator_macos(monkeypatch):
    if platform.system() == "Darwin":
        # Mock get_window_by_title
        with mock.patch("owa.env.window.callables.get_window_by_title") as mock_get_window_by_title:
            mock_window_info = WindowInfo(title="Test Window", rect=(100, 100, 500, 400), hWnd=123456)
            mock_get_window_by_title.return_value = mock_window_info

            # Mock CGWindowListCopyWindowInfo
            mock_window_info_list = [{"kCGWindowLayer": 0, "kCGWindowNumber": 123456}]
            with mock.patch("Quartz.CGWindowListCopyWindowInfo", return_value=mock_window_info_list):

                @callables.when_active("Test")
                def sample_function():
                    return "Function Executed"

                result = sample_function()
                assert result == "Function Executed"
