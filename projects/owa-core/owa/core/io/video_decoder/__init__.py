from owa.core.utils.typing import PathLike

from .pyav_decoder import PyAVVideoDecoder

# Conditional import with graceful fallback for optional dependency
try:
    from .torchcodec_decoder import TorchCodecVideoDecoder

    __all__ = ["PyAVVideoDecoder", "TorchCodecVideoDecoder"]
except ImportError:
    # Provide informative error when TorchCodec is unavailable
    class TorchCodecVideoDecoder:
        """Placeholder for TorchCodec decoder when dependency is not installed."""

        def __init__(self, source: PathLike, **kwargs):
            raise ImportError("TorchCodec is not available. Please install it with: pip install torchcodec>=0.4.0")

        def __new__(cls, source: PathLike, **kwargs):
            raise ImportError("TorchCodec is not available. Please install it with: pip install torchcodec>=0.4.0")

    __all__ = ["PyAVVideoDecoder", "TorchCodecVideoDecoder"]
