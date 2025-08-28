from typing import ClassVar

try:
    from torchcodec.decoders import VideoDecoder

    TorchCodecAvailable = True
except ImportError:
    TorchCodecAvailable = False
    VideoDecoder = None

from owa.core.utils.resource_cache import ResourceCache
from owa.core.utils.typing import PathLike


class TorchCodecVideoDecoder(VideoDecoder):
    """VideoDecoder that caches instances by path."""

    cache: ClassVar[ResourceCache] = ResourceCache(max_size=10)
    _skip_init = False

    def __new__(cls, source: PathLike, **kwargs):
        cache_key = str(source)
        if cache_key in cls.cache:
            instance = cls.cache[cache_key].obj
            instance._skip_init = True  # Set a flag before __init__
        else:
            instance = super().__new__(cls)
            instance._skip_init = False
        return instance

    def __init__(self, source: PathLike, **kwargs):
        if getattr(self, "_skip_init", False):
            return
        super().__init__(str(source), **kwargs)
        self._cache_key = str(source)
        # TorchCodec does not have a context manager protocol, so we use a no-op cleanup
        self.cache.add_entry(self._cache_key, self, lambda: None)

    def __exit__(self, exc_type, exc_value, traceback):
        self.cache.release_entry(self._cache_key)
