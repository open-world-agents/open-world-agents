"""Tests for trim command."""

from pathlib import Path
from unittest.mock import Mock, patch


from owa.cli.mcap import app as mcap_app
from owa.cli.mcap.trim import (
    default_mkv_namer,
    find_all_mkvs,
    get_duration,
    get_video_start_utc,
)


# === Unit Tests ===
class TestDefaultMkvNamer:
    def test_single_mkv_uses_mcap_stem(self, tmp_path):
        src_mkvs = {"video.mkv": tmp_path / "video.mkv"}
        dst_mcap = tmp_path / "output.mcap"
        namer = default_mkv_namer(src_mkvs, dst_mcap)
        result = namer(tmp_path / "video.mkv", dst_mcap)
        assert result == tmp_path / "output.mkv"

    def test_multiple_mkvs_use_original_stem_with_cut(self, tmp_path):
        src_mkvs = {
            "video1.mkv": tmp_path / "video1.mkv",
            "video2.mkv": tmp_path / "video2.mkv",
        }
        dst_mcap = tmp_path / "output.mcap"
        namer = default_mkv_namer(src_mkvs, dst_mcap)
        result = namer(tmp_path / "video1.mkv", dst_mcap)
        assert result == tmp_path / "video1_cut.mkv"


class TestFindAllMkvs:
    def test_find_mkvs_from_mcap(self, tmp_path):
        mcap_path = tmp_path / "test.mcap"
        mkv_path = tmp_path / "video.mkv"
        mkv_path.touch()

        mock_msg = Mock()
        mock_msg.decoded = Mock()
        mock_msg.decoded.media_ref = Mock()
        mock_msg.decoded.media_ref.uri = "video.mkv"

        with patch("owa.cli.mcap.trim.OWAMcapReader") as mock_reader:
            mock_reader.return_value.__enter__.return_value.iter_messages.return_value = [mock_msg]
            result = find_all_mkvs(mcap_path)

        assert "video.mkv" in result
        assert result["video.mkv"] == mkv_path.resolve()

    def test_find_mkvs_empty_when_no_screen_messages(self, tmp_path):
        mcap_path = tmp_path / "test.mcap"

        with patch("owa.cli.mcap.trim.OWAMcapReader") as mock_reader:
            mock_reader.return_value.__enter__.return_value.iter_messages.return_value = []
            result = find_all_mkvs(mcap_path)

        assert result == {}


class TestGetVideoStartUtc:
    def test_parse_subtitle_utc(self):
        # Mock ffmpeg output with subtitle containing UTC timestamp
        srt_output = "1\n00:00:00,000 --> 00:00:01,000\n1704067200000000000\n"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout=srt_output)
            result = get_video_start_utc(Path("test.mkv"))

        assert result == 1704067200000000000  # UTC in nanoseconds

    def test_returns_none_on_ffmpeg_error(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=1, stdout="")
            result = get_video_start_utc(Path("test.mkv"))

        assert result is None


class TestGetDuration:
    def test_parse_duration(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="123.456\n")
            result = get_duration(Path("test.mkv"))

        assert result == 123.456


# === CLI Tests ===
def test_trim_file_not_found(tmp_path, cli_runner):
    """Test error when input file doesn't exist."""
    result = cli_runner.invoke(
        mcap_app,
        ["trim", str(tmp_path / "nonexistent.mcap"), str(tmp_path / "output.mcap"), "--start", "10", "--duration", "30"],
    )
    assert result.exit_code == 1
    assert "not found" in result.output


def test_trim_success(tmp_path, cli_runner):
    """Test successful trim operation."""
    input_mcap = tmp_path / "input.mcap"
    input_mcap.touch()
    output_mcap = tmp_path / "output.mcap"

    with patch("owa.cli.mcap.trim.trim_recording") as mock_trim:
        mock_trim.return_value = (
            (0.5, 0.5),  # (before, after) margins
            {"video.mkv": tmp_path / "video.mkv"},  # src_mkvs
            {"video.mkv": tmp_path / "output.mkv"},  # dst_mkvs
        )
        result = cli_runner.invoke(
            mcap_app, ["trim", str(input_mcap), str(output_mcap), "--start", "10", "--duration", "30"]
        )

    assert result.exit_code == 0
    assert "Input:" in result.output
    assert "Trim range:" in result.output
    assert "Actual range:" in result.output
    assert "Output:" in result.output


def test_trim_with_custom_max_margin(tmp_path, cli_runner):
    """Test trim with custom max-margin option."""
    input_mcap = tmp_path / "input.mcap"
    input_mcap.touch()
    output_mcap = tmp_path / "output.mcap"

    with patch("owa.cli.mcap.trim.trim_recording") as mock_trim:
        mock_trim.return_value = (
            (1.0, 1.0),
            {"video.mkv": tmp_path / "video.mkv"},
            {"video.mkv": tmp_path / "output.mkv"},
        )
        result = cli_runner.invoke(
            mcap_app,
            ["trim", str(input_mcap), str(output_mcap), "--start", "10", "--duration", "30", "--max-margin", "10.0"],
        )
        mock_trim.assert_called_once()
        assert mock_trim.call_args[1]["max_margin"] == 10.0

    assert result.exit_code == 0


def test_trim_error_handling(tmp_path, cli_runner):
    """Test error handling when trim_recording raises an error."""
    input_mcap = tmp_path / "input.mcap"
    input_mcap.touch()
    output_mcap = tmp_path / "output.mcap"

    with patch("owa.cli.mcap.trim.trim_recording") as mock_trim:
        mock_trim.side_effect = ValueError("No MKV files found")
        result = cli_runner.invoke(
            mcap_app, ["trim", str(input_mcap), str(output_mcap), "--start", "10", "--duration", "30"]
        )

    assert result.exit_code == 1
    assert "Error:" in result.output

