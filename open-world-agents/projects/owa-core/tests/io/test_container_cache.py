"""
Comprehensive test suite for ContainerCache class.

Tests edge cases, concurrent access patterns, and bug fixes.
"""

import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from owa.core.io.video import (
    ContainerCache,
    VideoWriter,
    close_all_containers,
    get_container_cache_stats,
)


@pytest.fixture
def temp_video():
    """Create a temporary test video file."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        video_path = Path(tmp.name)

    # Create a simple test video
    with VideoWriter(video_path, fps=30.0) as writer:
        for i in range(10):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            frame[:, :, 0] = i * 25  # Simple pattern
            writer.write_frame(frame)

    yield video_path

    # Cleanup
    if video_path.exists():
        video_path.unlink()


@pytest.fixture
def cache():
    """Create a fresh ContainerCache instance for testing."""
    cache = ContainerCache(max_size=3, inactive_timeout=1.0)
    yield cache
    cache.close_all()


class TestContainerCache:
    """Test suite for ContainerCache class."""

    def test_basic_caching(self, cache, temp_video):
        """Test basic container caching functionality."""
        # First access should create new container
        container1 = cache.get_container(temp_video, mode="r")
        assert container1 is not None

        # Second access should reuse cached container
        container2 = cache.get_container(temp_video, mode="r")
        assert container2 is container1

        # Check cache stats
        stats = cache.get_cache_stats()
        assert stats["cache_size"] == 1
        assert stats["total_references"] == 2

    def test_reference_counting(self, cache, temp_video):
        """Test reference counting works correctly."""
        # Get container twice
        container1 = cache.get_container(temp_video, mode="r")
        container2 = cache.get_container(temp_video, mode="r")
        assert container1 is container2

        stats = cache.get_cache_stats()
        assert stats["total_references"] == 2

        # Release once
        cache.release_container(temp_video)
        stats = cache.get_cache_stats()
        assert stats["total_references"] == 1

        # Release again
        cache.release_container(temp_video)
        stats = cache.get_cache_stats()
        assert stats["total_references"] == 0

    def test_cache_eviction_when_full(self, cache, temp_video):
        """Test cache eviction when max_size is reached."""
        # Create multiple temp videos
        videos = []
        for i in range(5):  # More than cache max_size (3)
            with tempfile.NamedTemporaryFile(suffix=f"_{i}.mp4", delete=False) as tmp:
                video_path = Path(tmp.name)
                with VideoWriter(video_path, fps=30.0) as writer:
                    frame = np.zeros((32, 32, 3), dtype=np.uint8)
                    writer.write_frame(frame)
                videos.append(video_path)

        try:
            # Fill cache
            containers = []
            for i, video in enumerate(videos[:3]):
                container = cache.get_container(video, mode="r")
                containers.append(container)
                cache.release_container(video)  # Make them unused

            # Adding 4th should evict oldest
            cache.get_container(videos[3], mode="r")
            stats = cache.get_cache_stats()
            assert stats["cache_size"] <= 3  # Should not exceed max_size

            # Adding 5th when all are in use should not cache
            for video in videos[:3]:
                cache.get_container(video, mode="r")  # Make them used again

            cache.get_container(videos[4], mode="r")
            stats = cache.get_cache_stats()
            assert stats["cache_size"] <= 3

        finally:
            # Cleanup
            for video in videos:
                if video.exists():
                    video.unlink()

    def test_force_close_container(self, cache, temp_video):
        """Test force closing containers."""
        container = cache.get_container(temp_video, mode="r")
        assert container is not None

        stats = cache.get_cache_stats()
        assert stats["cache_size"] == 1

        # Force close should remove from cache
        cache.force_close_container(temp_video)
        stats = cache.get_cache_stats()
        assert stats["cache_size"] == 0

    def test_cleanup_inactive_containers(self, cache, temp_video):
        """Test cleanup of inactive containers."""
        # Set very short timeout for testing
        cache.inactive_timeout = 0.1

        cache.get_container(temp_video, mode="r")
        cache.release_container(temp_video)  # Make it unused

        # Wait for timeout
        time.sleep(0.2)

        # Next access should trigger cleanup
        cache.get_container(temp_video, mode="r")
        # The old container should have been cleaned up and new one created

    def test_write_mode_not_cached(self, cache, temp_video):
        """Test that write mode containers are not cached."""
        container = cache.get_container(temp_video, mode="w")
        assert container is not None

        stats = cache.get_cache_stats()
        assert stats["cache_size"] == 0  # Write mode not cached

    def test_concurrent_access(self, cache, temp_video):
        """Test thread-safe concurrent access."""
        results = []
        errors = []

        def worker():
            try:
                for _ in range(10):
                    container = cache.get_container(temp_video, mode="r")
                    results.append(id(container))
                    cache.release_container(temp_video)
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Should have no errors
        assert len(errors) == 0

        # All should get the same container (same ID)
        assert len(set(results)) == 1

    def test_error_handling_invalid_path(self, cache):
        """Test error handling for invalid paths."""
        with pytest.raises(Exception):  # Should raise some kind of error
            cache.get_container("/nonexistent/path.mp4", mode="r")

    def test_release_nonexistent_container(self, cache):
        """Test releasing a container that doesn't exist in cache."""
        # Should not raise an error, just log a warning
        cache.release_container("/nonexistent/path.mp4")

    def test_cache_stats_accuracy(self, cache, temp_video):
        """Test that cache statistics are accurate."""
        initial_stats = cache.get_cache_stats()
        assert initial_stats["cache_size"] == 0
        assert initial_stats["total_references"] == 0
        assert initial_stats["unused_containers"] == 0

        # Add container
        container = cache.get_container(temp_video, mode="r")
        stats = cache.get_cache_stats()
        assert stats["cache_size"] == 1
        assert stats["total_references"] == 1
        assert stats["unused_containers"] == 0

        # Reuse container
        container2 = cache.get_container(temp_video, mode="r")
        assert container2 is container
        stats = cache.get_cache_stats()
        assert stats["total_references"] == 2

        # Release container
        cache.release_container(temp_video)
        cache.release_container(temp_video)
        stats = cache.get_cache_stats()
        assert stats["cache_size"] == 1
        assert stats["total_references"] == 0
        assert stats["unused_containers"] == 1


def test_global_cache_functions(temp_video):
    """Test module-level cache functions."""
    # Clear any existing cache
    close_all_containers()

    initial_stats = get_container_cache_stats()
    assert initial_stats["cache_size"] == 0

    # These functions should work without errors
    from owa.core.io.video import get_video_container, release_video_container

    container = get_video_container(temp_video)
    assert container is not None

    stats = get_container_cache_stats()
    assert stats["cache_size"] == 1

    release_video_container(temp_video)
    close_all_containers()
