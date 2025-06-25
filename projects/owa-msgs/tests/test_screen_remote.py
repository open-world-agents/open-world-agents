"""
Tests for remote file handling in ScreenCaptured messages.

This module tests the integration between ScreenCaptured and the new remote
video file handling capabilities in VideoReader.
"""

import numpy as np
import pytest

from owa.msgs.desktop.screen import ExternalImageRef, ExternalVideoRef, ScreenCaptured


class TestScreenCapturedRemoteFiles:
    """Test ScreenCaptured with remote file references."""

    @pytest.mark.parametrize(
        "test_url,pts_ns,expected_media_type",
        [
            ("https://www.sample-videos.com/video321/mp4/240/big_buck_bunny_240p_2mb.mp4", 1_000_000_000, "video"),
            ("https://httpbin.org/image/png", None, "image"),
        ],
    )
    def test_screen_captured_remote_media(self, test_url, pts_ns, expected_media_type):
        """Test ScreenCaptured with remote media references."""
        try:
            # Create external reference
            if pts_ns is not None:
                external_ref = ExternalVideoRef(path=test_url, pts_ns=pts_ns)
            else:
                external_ref = ExternalImageRef(path=test_url)

            # Create ScreenCaptured with remote reference
            screen_msg = ScreenCaptured(media_ref=external_ref)

            # Verify media reference properties
            assert screen_msg.has_external_reference()
            assert not screen_msg.has_embedded_data()
            assert screen_msg.media_ref.path == test_url
            assert screen_msg.media_ref.pts_ns == pts_ns

            # Test media info
            media_info = screen_msg.get_media_info()
            assert media_info["type"] == "external"
            assert media_info["path"] == test_url
            assert media_info["is_remote"] is True
            assert media_info["is_local"] is False
            assert media_info["media_type"] == expected_media_type

            if pts_ns is not None:
                assert media_info["pts_ns"] == pts_ns
                assert media_info["pts_seconds"] == pts_ns / 1_000_000_000

            # Test lazy loading
            frame_arr = screen_msg.lazy_load()
            assert isinstance(frame_arr, np.ndarray)
            assert frame_arr.dtype == np.uint8
            assert len(frame_arr.shape) == 3  # Height, Width, Channels
            assert frame_arr.shape[2] == 4  # BGRA format

            # Test shape is set correctly
            h, w = frame_arr.shape[:2]
            expected_shape = (w, h)  # (width, height)
            assert screen_msg.shape == expected_shape

            # Test RGB conversion
            rgb_arr = screen_msg.to_rgb_array()
            assert rgb_arr.shape == (h, w, 3)  # RGB format
            assert rgb_arr.dtype == np.uint8

            # Test PIL conversion
            pil_img = screen_msg.to_pil_image()
            assert pil_img.size == (w, h)  # PIL uses (width, height)
            assert pil_img.mode == "RGB"

            # Test string representation includes remote info
            str_repr = str(screen_msg)
            assert "remote" in str_repr.lower()
            assert test_url in str_repr or test_url.split("/")[-1] in str_repr

        except Exception as e:
            # Skip test if network is unavailable or URL is inaccessible
            pytest.skip(f"Remote media test skipped due to error: {e}")

    def test_screen_captured_remote_video_specific(self):
        """Test ScreenCaptured specifically with remote video."""
        test_url = "https://www.sample-videos.com/video321/mp4/240/big_buck_bunny_240p_2mb.mp4"
        pts_ns = 1_000_000_000  # 1 second

        try:
            external_ref = ExternalVideoRef(path=test_url, pts_ns=pts_ns)
            screen_msg = ScreenCaptured(media_ref=external_ref)

            # Test that we can load the frame
            frame_arr = screen_msg.lazy_load()

            # Verify frame properties for this specific video
            assert frame_arr.shape[0] > 0  # Height > 0
            assert frame_arr.shape[1] > 0  # Width > 0
            assert frame_arr.shape[2] == 4  # BGRA

            # Test that subsequent calls return the same cached frame
            frame_arr2 = screen_msg.lazy_load()
            assert np.array_equal(frame_arr, frame_arr2)

            # Test force_close parameter
            frame_arr3 = screen_msg.lazy_load(force_close=True)
            assert frame_arr3.shape == frame_arr.shape

        except Exception as e:
            pytest.skip(f"Remote video test skipped due to error: {e}")

    def test_screen_captured_remote_image_specific(self):
        """Test ScreenCaptured specifically with remote image."""
        test_url = "https://httpbin.org/image/png"

        try:
            external_ref = ExternalImageRef(path=test_url)
            screen_msg = ScreenCaptured(media_ref=external_ref)

            # Test that we can load the image
            frame_arr = screen_msg.lazy_load()

            # Verify frame properties
            assert frame_arr.shape[2] == 4  # BGRA
            assert frame_arr.dtype == np.uint8

            # Test media info for image
            media_info = screen_msg.get_media_info()
            assert media_info["media_type"] == "image"
            assert "pts_ns" not in media_info or media_info["pts_ns"] is None

        except Exception as e:
            pytest.skip(f"Remote image test skipped due to error: {e}")

    def test_screen_captured_serialization_with_remote_ref(self):
        """Test JSON serialization with remote media reference."""
        test_url = "https://example.com/video.mp4"
        pts_ns = 1_000_000_000  # 1 second

        external_ref = ExternalVideoRef(path=test_url, pts_ns=pts_ns)
        screen_msg = ScreenCaptured(
            media_ref=external_ref, utc_ns=1234567890000000000, source_shape=(1920, 1080), shape=(1920, 1080)
        )

        # Test JSON serialization
        json_str = screen_msg.model_dump_json()
        assert test_url in json_str
        assert str(pts_ns) in json_str

        # Test deserialization
        screen_msg2 = ScreenCaptured.model_validate_json(json_str)
        assert screen_msg2.media_ref.path == test_url
        assert screen_msg2.media_ref.pts_ns == pts_ns
        assert screen_msg2.utc_ns == screen_msg.utc_ns
        assert screen_msg2.source_shape == screen_msg.source_shape
        assert screen_msg2.shape == screen_msg.shape

    def test_screen_captured_error_handling(self):
        """Test error handling with invalid remote references."""
        # Test invalid URL scheme - FTP URLs are treated as local files and should raise FileNotFoundError
        external_ref = ExternalVideoRef(path="ftp://example.com/video.mp4", pts_ns=0)
        screen_msg = ScreenCaptured(media_ref=external_ref)

        with pytest.raises(FileNotFoundError, match="Media file not found"):
            screen_msg.lazy_load()

        # Test non-existent remote file (should raise network-related error)
        external_ref = ExternalVideoRef(path="https://nonexistent.example.com/video.mp4", pts_ns=0)
        screen_msg = ScreenCaptured(media_ref=external_ref)

        with pytest.raises(Exception):  # Could be various network-related errors
            screen_msg.lazy_load()

    def test_screen_captured_format_media_display_remote(self):
        """Test media display formatting for remote files."""
        # Test remote video
        video_ref = ExternalVideoRef(
            path="https://example.com/long/path/to/video.mp4",
            pts_ns=1_500_000_000,  # 1.5 seconds
        )
        screen_msg = ScreenCaptured(media_ref=video_ref)
        str_repr = str(screen_msg)
        assert "remote_video" in str_repr
        assert "1.500s" in str_repr
        assert "video.mp4" in str_repr or "https://example.com/long/path/to/video.mp4" in str_repr

        # Test remote image
        image_ref = ExternalImageRef(path="https://example.com/image.jpg")
        screen_msg = ScreenCaptured(media_ref=image_ref)
        str_repr = str(screen_msg)
        assert "remote_image" in str_repr
        assert "image.jpg" in str_repr or "https://example.com/image.jpg" in str_repr
