from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class SamplingStrategy:
    window_start: float  # Relative to now (e.g., -0.25)
    window_end: float  # Relative to now (e.g., 0 or 0.25 for label)
    sample_interval: float  # 0 means discrete (event-based); >0 is uniform stride
    topic: str  # Sensor/stream/topic name (e.g., "screen")
    selector: Optional[Callable[[List], List]] = None
    # Allows custom selection policy (random, first-k, most recent...)


@dataclass
class PerceptionSamplingSpec:
    strategies: List[SamplingStrategy] = field(default_factory=list)
    # Optionally, you could make this a dict of topic -> strategy for faster lookup if needed
