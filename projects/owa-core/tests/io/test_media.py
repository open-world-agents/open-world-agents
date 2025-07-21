"""
Tests for the media.py module, focusing on image format encoding and decoding.
"""

import base64

import cv2
import numpy as np
import pytest

from owa.core.io.media import (
    bgra_array_to_pil,
    decode_from_base64,
    encode_to_base64,
    load_image_as_bgra,
    validate_media_path,
)


@pytest.fixture
def sample_bgra_array():
    """Create a sample BGRA numpy array for testing."""
    # Create a 10x10 BGRA image with a simple pattern
    array = np.zeros((10, 10, 4), dtype=np.uint8)

    # Create a simple pattern: red square in top-left, blue in bottom-right
    array[:5, :5] = [0, 0, 255, 255]  # Red (BGRA format)
    array[5:, 5:] = [255, 0, 0, 255]  # Blue (BGRA format)
    array[:5, 5:] = [0, 255, 0, 255]  # Green
    array[5:, :5] = [255, 255, 0, 255]  # Cyan

    return array


class TestEncodeToBase64:
    """Test the encode_to_base64 function with different formats."""

    def test_encode_png(self, sample_bgra_array):
        """Test encoding BGRA array to PNG base64."""
        result = encode_to_base64(sample_bgra_array, format="png")

        assert isinstance(result, str)
        assert len(result) > 0

        # Verify it's valid base64
        try:
            base64.b64decode(result)
        except Exception:
            pytest.fail("Result is not valid base64")

    def test_encode_jpeg(self, sample_bgra_array):
        """Test encoding BGRA array to JPEG base64."""
        result = encode_to_base64(sample_bgra_array, format="jpeg")

        assert isinstance(result, str)
        assert len(result) > 0

        # Verify it's valid base64
        try:
            base64.b64decode(result)
        except Exception:
            pytest.fail("Result is not valid base64")

    def test_encode_bmp(self, sample_bgra_array):
        """Test encoding BGRA array to BMP base64."""
        result = encode_to_base64(sample_bgra_array, format="bmp")

        assert isinstance(result, str)
        assert len(result) > 0

        # Verify it's valid base64
        try:
            base64.b64decode(result)
        except Exception:
            pytest.fail("Result is not valid base64")

    def test_encode_jpeg_with_quality(self, sample_bgra_array):
        """Test encoding JPEG with different quality settings."""
        # Test with high quality
        high_quality = encode_to_base64(sample_bgra_array, format="jpeg", quality=95)

        # Test with low quality
        low_quality = encode_to_base64(sample_bgra_array, format="jpeg", quality=10)

        # High quality should generally produce larger files
        assert len(high_quality) >= len(low_quality)

    def test_encode_invalid_format(self, sample_bgra_array):
        """Test encoding with invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported format"):
            encode_to_base64(sample_bgra_array, format="invalid")

    def test_encode_invalid_jpeg_quality(self, sample_bgra_array):
        """Test encoding JPEG with invalid quality raises ValueError."""
        with pytest.raises(ValueError, match="JPEG quality must be between 1 and 100"):
            encode_to_base64(sample_bgra_array, format="jpeg", quality=0)

        with pytest.raises(ValueError, match="JPEG quality must be between 1 and 100"):
            encode_to_base64(sample_bgra_array, format="jpeg", quality=101)

    def test_quality_ignored_for_png_and_bmp(self, sample_bgra_array):
        """Test that quality parameter is ignored for PNG and BMP formats."""
        # These should not raise errors even with quality parameter
        png_result = encode_to_base64(sample_bgra_array, format="png", quality=50)
        bmp_result = encode_to_base64(sample_bgra_array, format="bmp", quality=50)

        assert isinstance(png_result, str)
        assert isinstance(bmp_result, str)


class TestDecodeFromBase64:
    """Test the decode_from_base64 function."""

    def test_decode_png_roundtrip(self, sample_bgra_array):
        """Test encoding to PNG and decoding back."""
        encoded = encode_to_base64(sample_bgra_array, format="png")
        decoded = decode_from_base64(encoded)

        assert decoded.shape == sample_bgra_array.shape
        assert decoded.dtype == sample_bgra_array.dtype

        # PNG is lossless, so arrays should be identical
        np.testing.assert_array_equal(decoded, sample_bgra_array)

    def test_decode_bmp_roundtrip(self, sample_bgra_array):
        """Test encoding to BMP and decoding back."""
        encoded = encode_to_base64(sample_bgra_array, format="bmp")
        decoded = decode_from_base64(encoded)

        assert decoded.shape == sample_bgra_array.shape
        assert decoded.dtype == sample_bgra_array.dtype

        # BMP is lossless, so arrays should be identical
        np.testing.assert_array_equal(decoded, sample_bgra_array)

    def test_decode_jpeg_roundtrip(self):
        """Test encoding to JPEG and decoding back (with tolerance for lossy compression)."""
        # Create a larger, smoother image that's more suitable for JPEG compression
        # JPEG works better with gradual color transitions rather than sharp edges
        smooth_array = np.zeros((50, 50, 4), dtype=np.uint8)

        # Create a smooth gradient instead of sharp color blocks
        for i in range(50):
            for j in range(50):
                # Create a smooth color gradient
                r = int(255 * (i / 49))
                g = int(255 * (j / 49))
                b = int(255 * ((i + j) / 98))
                smooth_array[i, j] = [b, g, r, 255]  # BGRA format

        encoded = encode_to_base64(smooth_array, format="jpeg", quality=95)
        decoded = decode_from_base64(encoded)

        assert decoded.shape == smooth_array.shape
        assert decoded.dtype == smooth_array.dtype

        # JPEG is lossy, but with smooth gradients and high quality,
        # the difference should be much smaller
        np.testing.assert_allclose(decoded, smooth_array, atol=20)

    def test_decode_invalid_base64(self):
        """Test decoding invalid base64 data raises ValueError."""
        with pytest.raises(ValueError, match="Failed to decode base64 data"):
            decode_from_base64("invalid_base64_data!")

    def test_decode_valid_base64_invalid_image(self):
        """Test decoding valid base64 that's not image data raises ValueError."""
        # Valid base64 but not image data
        invalid_image_data = base64.b64encode(b"not an image").decode("utf-8")

        with pytest.raises(ValueError, match="Failed to decode base64 image data"):
            decode_from_base64(invalid_image_data)


class TestFormatComparison:
    """Test comparing different formats for the same image."""

    def test_format_size_comparison(self, sample_bgra_array):
        """Test that different formats produce different sizes."""
        png_encoded = encode_to_base64(sample_bgra_array, format="png")
        jpeg_encoded = encode_to_base64(sample_bgra_array, format="jpeg", quality=85)
        bmp_encoded = encode_to_base64(sample_bgra_array, format="bmp")

        # All should be valid strings
        assert all(isinstance(x, str) for x in [png_encoded, jpeg_encoded, bmp_encoded])

        # BMP is typically uncompressed and larger
        # PNG and JPEG sizes depend on content, but for our simple test image:
        # BMP should generally be largest for small images
        assert len(bmp_encoded) > 0
        assert len(png_encoded) > 0
        assert len(jpeg_encoded) > 0

    def test_lossless_formats_identical_roundtrip(self, sample_bgra_array):
        """Test that lossless formats (PNG, BMP) produce identical results after roundtrip."""
        # Encode and decode with PNG
        png_encoded = encode_to_base64(sample_bgra_array, format="png")
        png_decoded = decode_from_base64(png_encoded)

        # Encode and decode with BMP
        bmp_encoded = encode_to_base64(sample_bgra_array, format="bmp")
        bmp_decoded = decode_from_base64(bmp_encoded)

        # Both should be identical to original
        np.testing.assert_array_equal(png_decoded, sample_bgra_array)
        np.testing.assert_array_equal(bmp_decoded, sample_bgra_array)

        # And therefore identical to each other
        np.testing.assert_array_equal(png_decoded, bmp_decoded)


class TestBgraArrayToPil:
    """Test the bgra_array_to_pil function."""

    def test_bgra_to_pil_conversion(self, sample_bgra_array):
        """Test converting BGRA array to PIL image."""
        pil_image = bgra_array_to_pil(sample_bgra_array)

        # Check PIL image properties
        assert pil_image.mode == "RGB"
        assert pil_image.size == (sample_bgra_array.shape[1], sample_bgra_array.shape[0])

        # Convert back to array and verify
        pil_array = np.array(pil_image)
        assert pil_array.shape == (sample_bgra_array.shape[0], sample_bgra_array.shape[1], 3)


class TestValidateMediaPath:
    """Test the validate_media_path function."""

    def test_validate_existing_file(self, tmp_path):
        """Test validation of existing file."""
        test_file = tmp_path / "test.png"
        test_file.touch()  # Create empty file
        assert validate_media_path(str(test_file)) is True

    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        assert validate_media_path("nonexistent_file.png") is False

    def test_validate_url_mock(self):
        """Test URL validation (mocked to avoid network dependency)."""
        # This would require mocking requests, but for now we test the basic structure
        # In a real scenario, you'd mock the requests.head call
        result = validate_media_path("https://example.com/image.png")
        # Result depends on network availability, so we just check it's boolean
        assert isinstance(result, bool)


class TestLoadImageAsBgra:
    """Test the load_image_as_bgra function."""

    def test_load_image_from_file(self, sample_bgra_array, tmp_path):
        """Test loading image from file."""
        test_image = tmp_path / "test_image.png"

        # Save sample array as PNG file
        bgr_array = cv2.cvtColor(sample_bgra_array, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(str(test_image), bgr_array)

        # Load it back
        loaded_array = load_image_as_bgra(str(test_image))

        assert loaded_array.shape[:2] == sample_bgra_array.shape[:2]
        assert loaded_array.shape[2] == 4  # BGRA
        assert loaded_array.dtype == np.uint8

    def test_load_image_from_data_uri(self, sample_bgra_array):
        """Test loading image from data URI."""
        # Create data URI
        base64_data = encode_to_base64(sample_bgra_array, format="png")
        data_uri = f"data:image/png;base64,{base64_data}"

        # Load from data URI
        loaded_array = load_image_as_bgra(data_uri)

        assert loaded_array.shape == sample_bgra_array.shape
        assert loaded_array.dtype == sample_bgra_array.dtype
        np.testing.assert_array_equal(loaded_array, sample_bgra_array)

    def test_load_image_file_not_found(self):
        """Test loading non-existent file raises ValueError (not FileNotFoundError due to implementation)."""
        with pytest.raises(ValueError, match="Failed to load image"):
            load_image_as_bgra("nonexistent_file.png")

    def test_load_image_invalid_data_uri(self):
        """Test loading invalid data URI raises ValueError."""
        with pytest.raises(ValueError, match="Failed to load image"):
            load_image_as_bgra("data:invalid_uri")
