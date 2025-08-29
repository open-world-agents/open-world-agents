from typing import ClassVar

from torchcodec.decoders import VideoDecoder

from owa.core.utils.resource_cache import ResourceCache
from owa.core.utils.typing import PathLike


class TorchCodecVideoDecoder(VideoDecoder):
    """Cached TorchCodec video decoder for efficient resource management."""

    cache: ClassVar[ResourceCache] = ResourceCache(max_size=10)
    _skip_init = False

    def __new__(cls, source: PathLike, **kwargs):
        """Create or retrieve cached decoder instance."""
        cache_key = str(source)
        if cache_key in cls.cache:
            instance = cls.cache[cache_key].obj
            instance._skip_init = True
        else:
            instance = super().__new__(cls)
            instance._skip_init = False
        return instance

    def __init__(self, source: PathLike, **kwargs):
        """Initialize decoder if not retrieved from cache."""
        if getattr(self, "_skip_init", False):
            return
        super().__init__(str(source), **kwargs)
        self._cache_key = str(source)
        # Register with cache using no-op cleanup (TorchCodec handles cleanup internally)
        self.cache.add_entry(self._cache_key, self, lambda: None)

    def __exit__(self, exc_type, exc_value, traceback):
        """Release cache reference when used as context manager."""
        self.cache.release_entry(self._cache_key)
