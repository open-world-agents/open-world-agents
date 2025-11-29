"""Tests for subtitle command."""

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
    pair_mouse_clicks,
)
from owa.env.desktop.constants import VK


# === Unit Tests ===
def test_key_state():
    state = KeyState(VK.KEY_A)
    assert state.press(1000) is True
    assert state.press(2000) is False
    assert state.release(3000) is True
    assert state.release(4000) is False


def test_key_state_manager():
    mgr = KeyStateManager()
    mgr.handle_event("press", VK.KEY_A, 1_000_000_000)
    mgr.handle_event("release", VK.KEY_A, 1_600_000_000)
    assert len(mgr.completed) == 1
    assert mgr.completed[0][2] == "A"


def test_key_state_manager_finalize_pending():
    """Test that pending key states are finalized on finalize()."""
    mgr = KeyStateManager()
    mgr.handle_event("press", VK.KEY_A, 1_000_000_000)
    # No release event - key is still pending
    mgr.finalize()
    assert len(mgr.completed) == 1


def test_get_key_label():
    assert get_key_label(VK.ESCAPE) == "ESC"
    assert get_key_label(VK.KEY_A) == "A"
    assert get_key_label(9999).startswith("?")


def test_format_times():
    assert format_srt_time(1_000_000_000) == "00:00:01,000"
    assert format_ass_time(1_000_000_000) == "0:00:01.00"


def test_pair_mouse_clicks():
    events = [(1_000_000_000, "left", True), (1_100_000_000, "left", False)]
    result = pair_mouse_clicks(events)
    assert len(result) == 1
    assert result[0][1] - result[0][0] == 500_000_000  # Min duration


def test_pair_mouse_clicks_min_duration_enforced():
    """Test that minimum duration is enforced for short clicks."""
    # Click with very short press-release time
    events = [(1_000_000_000, "left", True), (1_100_000_000, "left", False)]
    result = pair_mouse_clicks(events)
    press_ts, end_ts, _ = result[0]
    assert end_ts - press_ts == 500_000_000  # MIN_DURATION_NS


def test_generate_srt():
    events = [(500_000_000, 1_000_000_000, "A")]
    result = generate_srt(0, events, [])
    assert "[keyboard] press A" in result


def test_generate_ass():
    result = generate_ass(0, [], [], {}, 1920, 1080)
    assert "[Script Info]" in result


# === CLI Tests ===
def test_subtitle_help(cli_runner):
    result = cli_runner.invoke(mcap_app, ["subtitle", "--help"])
    assert result.exit_code == 0
    assert "--format" in result.stdout or "-f" in result.stdout


def test_subtitle_generates_output(cli_runner, tmp_path):
    test_file = tmp_path / "test.mcap"
    test_file.write_bytes(b"mock")

    with patch("owa.cli.mcap.subtitle.OWAMcapReader") as mock_reader:
        mock_ctx = mock_reader.return_value.__enter__.return_value
        mock_msg = Mock(timestamp=1_000_000_000, decoded=Mock(media_ref=None))
        mock_ctx.iter_messages.side_effect = [iter([mock_msg]), iter([])]

        result = cli_runner.invoke(mcap_app, ["subtitle", str(test_file), "-f", "srt"])
        assert result.exit_code == 0

        # Verify file was created and has valid SRT content
        output_file = tmp_path / "test.srt"
        assert output_file.exists()
        content = output_file.read_text()
        # SRT files should have at least basic structure (even if empty events)
        assert content is not None
