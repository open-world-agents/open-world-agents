"""
MCAP Event Sampling Specification

This module defines a flexible framework for sampling events from MCAP files,
which contain (timestamp, topic, msg) triplets. It supports two main types of events:

1. Discrete events: Events that occur at specific moments without continuity
   (e.g., keyboard presses, button clicks)
   - Sampling patterns: All events in window, first K events, last K events

2. Continuous events: Events with temporal continuity where sampling rate matters
   (e.g., camera frames, sensor readings)
   - Sampling patterns: All events in window, regular sampling by FPS,
     with optional interpolation
"""

from enum import Enum
from typing import Any, Callable, List, Literal, Optional, TypeAlias, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EventType(str, Enum):
    """Defines the type of event stream based on its temporal characteristics."""

    # Events that occur at discrete points in time (e.g., keyboard/mouse inputs)
    DISCRETE = "discrete"
    # Events that represent continuous data streams (e.g., video, sensor readings)
    CONTINUOUS = "continuous"


class BaseSamplingStrategy(BaseModel):
    """
    Base class for all sampling strategies.

    Defines the common parameters needed for any type of event sampling,
    including the topic to sample and the time window relative to 'now'.
    """

    topic: str = Field(..., description="Sensor/stream/topic name to sample from")
    msg_filter: Optional[Callable[[Any], bool]] = Field(
        None, description="Function to filter messages based on their content"
    )
    window_start: float = Field(..., description="Start time relative to 'now' (e.g., -0.25 seconds)")
    window_end: float = Field(..., description="End time relative to 'now' (e.g., 0 seconds)")

    @property
    def requires_future_info(self) -> bool:
        """Check if the sampling strategy requires future information."""
        return self.window_start > 0

    @model_validator(mode="after")
    def check_window(self) -> "BaseSamplingStrategy":
        """Ensure that window_start is less than window_end."""
        if self.window_start >= self.window_end:
            raise ValueError(f"window_start {self.window_start} must be less than window_end {self.window_end}")
        return self

    # Disallow extra fields. Since BaseSamplingStrategy is a base class,
    # it's required to prevent confusion between the subclasses.
    model_config = ConfigDict(extra="forbid")


class DiscreteSamplingStrategy(BaseSamplingStrategy):
    """
    Sampling strategy for discrete events.

    Suitable for events that occur at specific moments without continuity,
    such as keyboard presses, mouse clicks, or other sporadic inputs.

    Sampling patterns:
    - all: Include all events within the time window
    - first_k: Include only the first K events within the window
    - last_k: Include only the last K events within the window
    """

    event_type: Literal[EventType.DISCRETE] = EventType.DISCRETE
    mode: Literal["all", "first_k", "last_k"] = Field(
        "all", description="Sampling mode: 'all' for all events, 'first_k'/'last_k' for a subset"
    )
    k: Optional[int] = Field(None, description="Number of events to select when mode is first_k or last_k")
    # Workaround for stateful topics that require the most recent state snapshot
    include_prior_state: bool = Field(
        False, description="Whether to include the most recent state snapshot before window_start"
    )
    state_topic: Optional[str] = Field(
        None, description="Topic containing state snapshots, if different from main topic"
    )
    state_update_fn: Optional[Callable[[Any], Any]] = Field(
        None, description="Function to update state message given the raw event message"
    )
    state_to_event_fn: Optional[Callable[[Any], Any]] = Field(
        None, description="Function to convert state message to a format suitable for the main topic"
    )

    @field_validator("k")
    def validate_k(cls, v, info):
        """Ensure k is provided when using first_k or last_k modes."""
        if info.data.get("mode") in ["first_k", "last_k"] and v is None:
            raise ValueError("k must be specified when mode is first_k or last_k")
        return v

    @model_validator(mode="after")
    def validate_include_prior_state(self) -> "DiscreteSamplingStrategy":
        """Ensure include_prior_state is set only if state_topic is provided and mode is not 'all'."""
        if self.include_prior_state and not self.state_topic:
            raise ValueError("include_prior_state requires a state_topic to be specified")
        if self.mode != "all" and self.include_prior_state:
            raise ValueError("include_prior_state is not applicable when mode is not 'all'")
        return self


class ContinuousSamplingStrategy(BaseSamplingStrategy):
    """
    Sampling strategy for continuous events.

    Suitable for events that represent continuous data streams where
    regular sampling is important, such as video frames, sensor readings,
    or telemetry data.

    Sampling patterns:
    - all: Include all events within the time window
    - first_k: Include only the first K events within the window
    - last_k: Include only the last K events within the window

    Additional control:
    - fps: Required parameter for rate-limited sampling (e.g., sample at 5 fps regardless of source rate)
    - interpolate: Whether to interpolate between samples if exact timestamps aren't available
    """

    event_type: Literal[EventType.CONTINUOUS] = EventType.CONTINUOUS
    mode: Literal["all", "first_k", "last_k"] = Field(
        "all", description="Sampling mode: 'all' for all events, 'first_k'/'last_k' for a subset"
    )
    k: Optional[int] = Field(None, description="Number of events to select when mode is first_k or last_k")
    fps: float = Field(
        ..., description="Frames per second for continuous sampling (required to control sampling rate)"
    )
    interpolate: bool = Field(
        False, description="Whether to interpolate between samples if exact timestamps aren't available"
    )
    interpolation_fn: Optional[Callable[[Any], Any]] = Field(
        None, description="Function to interpolate between samples if interpolate is True"
    )

    @field_validator("k")
    def validate_k(cls, v, info):
        """Ensure k is provided when using first_k or last_k modes."""
        if info.data.get("mode") in ["first_k", "last_k"] and v is None:
            raise ValueError("k must be specified when mode is first_k or last_k")
        return v

    @model_validator(mode="after")
    def validate_interpolate(self):
        """Ensure interpolation function is provided if interpolate is True."""
        if self.interpolate and self.interpolation_fn is None:
            raise ValueError("interpolation_fn must be specified when interpolate is True")
        return self


# Union type for any valid sampling strategy
SamplingStrategy: TypeAlias = Union[DiscreteSamplingStrategy, ContinuousSamplingStrategy]


class PerceptionSamplingSpec(dict[str, SamplingStrategy]):
    """
    Complete specification for sampling events from an MCAP file.

    This class combines multiple sampling strategies to define how
    different types of events should be sampled from the MCAP file.

    Usage:
    1. Create sampling strategies for each topic/event type
    2. Combine them in a PerceptionSamplingSpec
    3. Use this spec to guide the actual sampling from the MCAP file

    Example:
    ```
    # Sample keyboard events (last 3 key presses)
    keyboard_strategy = DiscreteSamplingStrategy(
        topic="keyboard",
        window_start=-5.0,  # 5 seconds before "now"
        window_end=0.0,     # up to "now"
        mode="last_k",
        k=3
    )

    # Sample screen frames at 5fps
    screen_strategy = ContinuousSamplingStrategy(
        topic="screen",
        window_start=-1.0,  # 1 second before "now"
        window_end=0.0,     # up to "now"
        fps=5.0             # Required: Sample at 5 frames per second
    )

    # Create the full spec
    spec = PerceptionSamplingSpec(
        inputs=[keyboard_strategy, screen_strategy]
    )
    ```
    """

    @property
    def start_time(self) -> float:
        """Earliest start time across all sampling strategies."""
        return min(strategy.window_start for strategy in self.values())

    @property
    def end_time(self) -> float:
        """Latest end time across all sampling strategies."""
        return max(strategy.window_end for strategy in self.values())

    @property
    def duration(self) -> float:
        """Total time length covered by the sampling strategies."""
        return self.end_time - self.start_time
