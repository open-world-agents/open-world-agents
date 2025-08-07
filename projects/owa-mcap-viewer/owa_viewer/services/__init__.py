"""
Services package for OWA MCAP Viewer.

This package contains all business logic services organized by domain:
- CacheService: Cache management and cleanup
- FileService: File operations and repository access
- McapService: MCAP file processing and data extraction
- ServiceContainer: Dependency injection and service coordination
"""

from .cache_service import CacheService
from .file_service import FileService
from .mcap_service import McapService


class ServiceContainer:
    """
    Container for managing service dependencies with proper initialization order.

    This follows the dependency injection pattern to ensure clean separation
    of concerns while providing a simple interface for accessing services.
    """

    def __init__(self):
        """Initialize services in proper dependency order"""
        # Core services (no dependencies)
        self.cache_service = CacheService()

        # File service (depends on cache service)
        self.file_service = FileService(self.cache_service)

        # MCAP service (depends on file and cache services)
        self.mcap_service = McapService(self.file_service, self.cache_service)

    async def start_periodic_cleanup(self) -> None:
        """Start background cleanup tasks"""
        await self.cache_service.start_periodic_cleanup()

    async def stop_periodic_cleanup(self) -> None:
        """Stop background cleanup tasks"""
        await self.cache_service.stop_periodic_cleanup()

    def clear_cache(self) -> None:
        """Clear all caches"""
        self.cache_service.clear_file_lists()


# Global service container instance
services = ServiceContainer()

# Export main classes and instance
__all__ = ["CacheService", "FileService", "McapService", "ServiceContainer", "services"]
