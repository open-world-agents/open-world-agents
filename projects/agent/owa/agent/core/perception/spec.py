from typing import Any, Callable, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BasePerceptionSpec(BaseModel):
    """
    Base class for all sampling strategies.

    Defines the common parameters needed for any type of event sampling,
    including the topic to sample and the time window relative to 'now'.
    """

    topics: list[str] = Field(..., description="Sensor/stream/topic name to sample from")
    window_start: float = Field(..., description="Start time relative to 'now' (e.g., -0.25 seconds)")
    window_end: float = Field(..., description="End time relative to 'now' (e.g., 0 seconds)")

    @property
    def requires_future_info(self) -> bool:
        """Check if the sampling strategy requires future information."""
        return self.window_start > 0

    @model_validator(mode="after")
    def check_window(self) -> "BasePerceptionSpec":
        """Ensure that window_start is less than window_end."""
        if self.window_start >= self.window_end:
            raise ValueError(f"window_start {self.window_start} must be less than window_end {self.window_end}")
        return self

    # Disallow extra fields. Since BasePerceptionSpec is a base class,
    # it's required to prevent confusion between the subclasses.
    model_config = ConfigDict(extra="forbid")


class SamplingConfig(BaseModel):
    """
    Configuration for sampling events from a (near-)continuous events.
    Only messages satisfying `sample_if` are modified by this config. Others remains same.
    If `sample_if` is None, all messages are sampled.

    Args:
        sampling_rate: Rate at which to sample messages (e.g., 5.0 for 5Hz).
        sample_if: Function to determine if a message should be sampled. If None, all messages are sampled.
        do_interpolate: Whether to interpolate between samples if exact timestamps aren't available.
        interpolation_fn: Function to interpolate between samples if do_interpolate is True.
    """

    sampling_rate: float = Field(..., description="Rate at which to sample messages (e.g., 5.0 for 5Hz)")
    sample_if: Optional[Callable[[Any], bool]] = Field(
        None, description="Function to determine if a message should be sampled. If None, all messages are sampled."
    )
    do_interpolate: bool = Field(
        False, description="Whether to interpolate between samples if exact timestamps aren't available"
    )
    interpolation_fn: Optional[Callable[[Any], Any]] = Field(
        None, description="Function to interpolate between samples if do_interpolate is True"
    )

    @model_validator(mode="after")
    def validate_interpolation_fn(self) -> "SamplingConfig":
        """Ensure interpolation function is provided if do_interpolate is True."""
        if self.do_interpolate and self.interpolation_fn is None:
            raise ValueError("interpolation_fn must be specified when do_interpolate is True")
        return self


class TrimConfig(BaseModel):
    """
    Configuration for trimming events, e.g. parse only the first or last K events within a time window.
    Only messages satisfying `trim_if` are modified by this config. Others remains same.
    If `trim_if` is None, all messages are trimmed.

    patterns:
    - first_k: Include only the first K events within the window
    - last_k: Include only the last K events within the window
    """

    trim_mode: Literal["first_k", "last_k"] = Field(
        ..., description="Trim mode: 'all' for all events, 'first_k'/'last_k' for a subset"
    )
    trim_if: Optional[Callable[[Any], bool]] = Field(
        None, description="Function to determine if a message should be trimmed. If None, all messages are trimmed."
    )
    trim_k: Optional[int] = Field(None, description="Number of events to trim when mode is first_k or last_k")


class PerceptionSpec(BasePerceptionSpec):
    """
    Configuration for sampling and trimming events from a stream.

    Processing pipeline:
        1. Apply `sample_configs` to filter and sample raw events.
        2. Then apply `trim_configs` to further refine the sampled events.

    Each PerceptionSpec applies to a single channel.
    """

    sample_configs: List[SamplingConfig] = Field(
        default_factory=list, description="List of sampling configurations for the event stream"
    )
    trim_configs: List[TrimConfig] = Field(
        default_factory=list, description="List of trimming configurations for the event stream"
    )


class PerceptionSpecDict(dict[str, PerceptionSpec]):
    """Dictionary-like object to hold multiple `PerceptionSpec` instances."""

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
