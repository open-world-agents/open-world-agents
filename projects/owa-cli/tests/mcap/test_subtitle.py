"""Tests for subtitle command functionality."""

from unittest.mock import Mock, patch


from owa.cli.mcap import app as mcap_app
from owa.cli.mcap.subtitle import (
    KeyState,
    KeyStateManager,
    format_ass_time,
    format_srt_time,
    generate_ass,
    generate_srt,
    get_key_label,
)
from owa.env.desktop.constants import VK


class TestKeyState:
    """Tests for KeyState class."""

    def test_initial_state(self):
        state = KeyState(VK.KEY_A)
        assert not state.is_pressed
        assert state.press_time is None

    def test_press_release(self):
        state = KeyState(VK.KEY_A)
        assert state.press(1000) is True
        assert state.is_pressed
        assert state.press(2000) is False  # Already pressed
        assert state.release(3000) is True
        assert not state.is_pressed
        assert state.release(4000) is False  # Already released

    def test_get_duration_min_duration(self):
        state = KeyState(VK.KEY_A)
        state.press(1_000_000_000)
        state.release(1_100_000_000)  # 100ms
        start, end = state.get_duration()
        assert start == 1_000_000_000
        assert end - start == 500_000_000  # Min 500ms


class TestKeyStateManager:
    """Tests for KeyStateManager class."""

    def test_single_key(self):
        mgr = KeyStateManager()
        mgr.handle_event("press", VK.KEY_A, 1_000_000_000)
        mgr.handle_event("release", VK.KEY_A, 1_600_000_000)
        assert len(mgr.completed) == 1
        assert mgr.completed[0][2] == "A"

    def test_finalize_pending(self):
        mgr = KeyStateManager()
        mgr.handle_event("press", VK.KEY_A, 1_000_000_000)
        mgr.finalize()
        assert len(mgr.completed) == 1


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_key_label_special(self):
        assert get_key_label(VK.ESCAPE) == "ESC"
        assert get_key_label(VK.RETURN) == "ENTER"
        assert get_key_label(VK.SPACE) == "SPACE"

    def test_get_key_label_regular(self):
        assert get_key_label(VK.KEY_A) == "A"
        assert get_key_label(VK.KEY_Z) == "Z"

    def test_get_key_label_unknown(self):
        assert get_key_label(9999).startswith("?")

    def test_format_srt_time(self):
        assert format_srt_time(0) == "00:00:00,000"
        assert format_srt_time(1_000_000_000) == "00:00:01,000"
        assert format_srt_time(3661_500_000_000) == "01:01:01,500"

    def test_format_ass_time(self):
        assert format_ass_time(0) == "0:00:00.00"
        assert format_ass_time(1_000_000_000) == "0:00:01.00"


class TestGenerateSrt:
    """Tests for SRT generation."""

    def test_empty(self):
        result = generate_srt(0, [], [])
        assert result == ""

    def test_keyboard_event(self):
        events = [(500_000_000, 1_000_000_000, "A")]
        result = generate_srt(0, events, [])
        assert "[keyboard] press A" in result
        assert "00:00:00,500" in result


class TestGenerateAss:
    """Tests for ASS generation."""

    def test_header(self):
        result = generate_ass(0, [], [], {}, 1920, 1080)
        assert "[Script Info]" in result
        assert "PlayResX: 1920" in result

    def test_keyboard_event(self):
        events = [(500_000_000, 1_000_000_000, "A")]
        result = generate_ass(0, events, [], {}, 1920, 1080)
        assert "KeyPressed" in result


class TestSubtitleCli:
    """Tests for subtitle CLI command."""

    def test_help(self, cli_runner):
        result = cli_runner.invoke(mcap_app, ["subtitle", "--help"])
        assert result.exit_code == 0
        assert "Generate subtitle file" in result.stdout

    def test_nonexistent_file(self, cli_runner, tmp_path):
        result = cli_runner.invoke(mcap_app, ["subtitle", str(tmp_path / "missing.mcap")])
        assert result.exit_code != 0

    def test_generates_output(self, cli_runner, tmp_path):
        test_file = tmp_path / "test.mcap"
        test_file.write_bytes(b"mock")

        with patch("owa.cli.mcap.subtitle.OWAMcapReader") as mock_reader:
            mock_ctx = mock_reader.return_value.__enter__.return_value
            mock_msg = Mock(timestamp=1_000_000_000)
            mock_msg.decoded = Mock(media_ref=None)
            mock_ctx.iter_messages.side_effect = [iter([mock_msg]), iter([])]

            result = cli_runner.invoke(mcap_app, ["subtitle", str(test_file), "-f", "srt"])
            assert result.exit_code == 0
            assert (tmp_path / "test.srt").exists()
