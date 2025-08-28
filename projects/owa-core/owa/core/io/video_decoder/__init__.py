from owa.core.utils.typing import PathLike

from .pyav_decoder import PyAVVideoDecoder

# Import TorchCodecVideoDecoder with graceful fallback
try:
    from .torchcodec_decoder import TorchCodecVideoDecoder

    __all__ = ["PyAVVideoDecoder", "TorchCodecVideoDecoder"]
except ImportError:
    # TorchCodec not available, provide a placeholder class that gives helpful error messages
    class TorchCodecVideoDecoder:
        """Placeholder class when TorchCodec is not available."""

        def __init__(self, source: PathLike, **kwargs):
            raise ImportError("TorchCodec is not available. Please install it with: pip install torchcodec>=0.4.0")

        def __new__(cls, source: PathLike, **kwargs):
            raise ImportError("TorchCodec is not available. Please install it with: pip install torchcodec>=0.4.0")

    __all__ = ["PyAVVideoDecoder", "TorchCodecVideoDecoder"]
