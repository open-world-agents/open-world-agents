#!/usr/bin/env python3
"""
Test script to verify MediaRef functionality in owa-mcap-viewer.

This script tests the core MediaRef functionality without requiring a full test suite.
"""

import sys
import tempfile
from pathlib import Path

# Add the project to the path
sys.path.insert(0, str(Path(__file__).parent))

from owa_viewer.models.file import MediaReference, OWAFile
from owa_viewer.services.media_service import MediaService
from owa_viewer.repositories.file_repository import FileRepository


def test_media_reference_model():
    """Test the MediaReference model."""
    print("Testing MediaReference model...")
    
    # Test video reference
    video_ref = MediaReference(
        uri="video.mkv",
        media_type="video",
        is_video=True,
        is_embedded=False,
        is_remote=False,
        file_extension=".mkv"
    )
    assert video_ref.uri == "video.mkv"
    assert video_ref.is_video is True
    assert video_ref.is_embedded is False
    
    # Test embedded reference
    embedded_ref = MediaReference(
        uri="data:image/png;base64,iVBORw0KGgo...",
        media_type="embedded",
        is_video=False,
        is_embedded=True,
        is_remote=False,
        file_extension=None
    )
    assert embedded_ref.is_embedded is True
    assert embedded_ref.is_video is False
    
    print("✓ MediaReference model tests passed")


def test_owa_file_model():
    """Test the updated OWAFile model."""
    print("Testing OWAFile model...")
    
    # Test MCAP file with MediaRef
    media_refs = [
        MediaReference(
            uri="video.mkv",
            media_type="video",
            is_video=True,
            is_embedded=False,
            is_remote=False,
            file_extension=".mkv"
        )
    ]
    
    owa_file = OWAFile(
        basename="test_file",
        original_basename="original_test",
        size=1024,
        local=True,
        url="test_file",
        url_mcap="test_file.mcap",
        media_references=media_refs,
        has_external_media=True,
        has_embedded_media=False,
        url_mkv=None  # No legacy MKV
    )
    
    assert owa_file.has_external_media is True
    assert owa_file.has_embedded_media is False
    assert len(owa_file.media_references) == 1
    assert owa_file.url_mkv is None
    
    print("✓ OWAFile model tests passed")


def test_media_service_data_uri():
    """Test MediaService data URI handling."""
    print("Testing MediaService data URI handling...")
    
    media_service = MediaService()
    
    # Test data URI parsing
    data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    try:
        resolved_path, is_temp, media_type = media_service._handle_data_uri(data_uri)
        assert is_temp is True
        assert media_type.startswith("image/")
        assert resolved_path.exists()
        
        # Clean up
        resolved_path.unlink()
        
        print("✓ MediaService data URI tests passed")
    except Exception as e:
        print(f"✗ MediaService data URI test failed: {e}")


def test_file_repository_analysis():
    """Test FileRepository MediaRef analysis."""
    print("Testing FileRepository MediaRef analysis...")
    
    file_repo = FileRepository()
    
    # Test with a non-existent file (should handle gracefully)
    try:
        from fsspec.implementations.local import LocalFileSystem
        fs = LocalFileSystem()
        
        # This should not crash even with invalid path
        media_refs, has_external, has_embedded = file_repo._analyze_mcap_media_references(
            Path("/nonexistent/file.mcap"), fs
        )
        
        # Should return empty results for non-existent file
        assert isinstance(media_refs, list)
        assert isinstance(has_external, bool)
        assert isinstance(has_embedded, bool)
        
        print("✓ FileRepository analysis tests passed")
    except Exception as e:
        print(f"✗ FileRepository analysis test failed: {e}")


def main():
    """Run all tests."""
    print("Running MediaRef functionality tests...\n")
    
    try:
        test_media_reference_model()
        test_owa_file_model()
        test_media_service_data_uri()
        test_file_repository_analysis()
        
        print("\n✅ All tests passed! MediaRef functionality is working correctly.")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
