"""Configuration classes for interval extractors."""

from dataclasses import dataclass, field
from typing import Any, Dict

from . import selector


@dataclass
class IntervalExtractorConfig:
    """Configuration for interval extractor.

    Uses class_name to dynamically instantiate the extractor class from the selector module.
    The kwargs are passed to the constructor of the specified class.
    """

    class_name: str = "All"
    kwargs: Dict[str, Any] = field(default_factory=lambda: {})

    def create_extractor(self):
        """Create the interval extractor instance from configuration."""
        if not hasattr(selector, self.class_name):
            raise ValueError(f"Unknown interval extractor class: {self.class_name}")

        extractor_class = getattr(selector, self.class_name)
        return extractor_class(**self.kwargs)
