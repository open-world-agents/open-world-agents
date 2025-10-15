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

    class_name: str = "InactivityFilter"
    kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"screen_inactivity_threshold": 1.0, "input_inactivity_threshold": 5.0}
    )

    def create_extractor(self):
        """Create the interval extractor instance from configuration."""
        if not hasattr(selector, self.class_name):
            raise ValueError(f"Unknown interval extractor class: {self.class_name}")

        extractor_class = getattr(selector, self.class_name)
        return extractor_class(**self.kwargs)
