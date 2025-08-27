from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Iterator, Self, Type, TypedDict

import numpy as np

from mcap_owa.highlevel import OWAMcapReader
from mcap_owa.highlevel.mcap_msg import McapMessage
from owa.core.utils.typing import PathLike


class EventType(StrEnum):
    """Type of event."""

    SCREEN = "screen"
    KEYBOARD = "keyboard"
    MOUSE_OP = "mouse_op"
    MOUSE_NOP = "mouse_nop"
    ALL = "all"


class PerEventMetric(ABC):
    """Base class for all metrics."""

    name: str
    event_type: EventType
    value: Any

    @abstractmethod
    def compute(self, src_event: McapMessage, dst_event: McapMessage) -> Self: ...

    @classmethod
    @abstractmethod
    def aggregate(cls, metrics: list[Self]) -> dict[str, Any]: ...

    def __repr__(self) -> str:
        return f"{self.name}: {self.value}"


class TimestampPE(PerEventMetric):
    name: str = "timestamp_pe"
    event_type: EventType = EventType.ALL

    def compute(self, src_event: McapMessage, dst_event: McapMessage) -> Self:
        self.value = (src_event.timestamp - dst_event.timestamp) / src_event.timestamp
        return self

    @classmethod
    def aggregate(cls, metrics: list[Self]) -> dict[str, Any]:
        values = np.asarray([m.value for m in metrics])
        q0, q1, q2, q3, q4 = np.percentile(values, [0, 25, 50, 75, 100])
        iqm = np.mean(values[(values >= q1) & (values <= q3)])

        return {
            "timestamp_pe_mean": np.mean(values),
            "timestamp_pe_std": np.std(values),
            "timestamp_pe_iqm": iqm,
            "timestamp_pe_q0": q0,
            "timestamp_pe_q1": q1,
            "timestamp_pe_q2": q2,
            "timestamp_pe_q3": q3,
            "timestamp_pe_q4": q4,
        }


metric_classes: list[Type[PerEventMetric]] = [TimestampPE]


class PerEventComparisonResult(TypedDict):
    """Result of comparing two events."""

    comparable: bool
    comparison_status: str

    metrics: dict[str, PerEventMetric]


class EvaluationResult(TypedDict):
    """Result of evaluation."""

    # We need:
    # - unweighted sum/mean/IQM/median of all per-event metrics
    # - weighted sum/mean/IQM/median of all per-event metrics (by position of event, EMA)


def _determine_event_type(event: McapMessage) -> EventType:
    """Determine the event type from a decoded event object."""
    event_type = event.topic
    # Handle mouse events with special categorization
    if event_type == "mouse/raw":
        if event.decoded.button_flags != 0:
            return EventType.MOUSE_OP
        else:
            return EventType.MOUSE_NOP
    # Handle other event types - can be extended here
    if event_type in ["keyboard", "screen"]:
        return EventType(event_type)
    raise ValueError(f"Unknown event type: {event_type}")


def compute_event_metric(src_event: McapMessage, dst_event: McapMessage) -> PerEventComparisonResult:
    """Compare two events."""
    result: dict = {"metrics": {}}
    src_event_type = _determine_event_type(src_event)
    dst_event_type = _determine_event_type(dst_event)
    if src_event_type != dst_event_type:
        result["comparable"] = False
        result["comparison_status"] = "type_mismatch"
        return PerEventComparisonResult(**result)

    result["comparable"] = True
    result["comparison_status"] = "valid"

    for metric_class in metric_classes:
        if metric_class.event_type != "all" and metric_class.event_type != src_event_type:
            continue
        metric = metric_class().compute(src_event, dst_event)
        result["metrics"][metric.name] = metric

    return PerEventComparisonResult(**result)


def compute_metrics(src_events: Iterator[McapMessage], dst_events: Iterator[McapMessage]) -> EvaluationResult:
    """Evaluate the accuracy of the agent's predictions."""
    per_event_results = []
    for src_event, dst_event in zip(src_events, dst_events):
        comparison_result = compute_event_metric(src_event, dst_event)
        per_event_results.append(comparison_result)

    # Aggregate metrics
    result = {}
    for metric_class in metric_classes:
        aggregated_metrics = metric_class.aggregate(
            [r["metrics"][metric_class.name] for r in per_event_results if metric_class.name in r["metrics"]]
        )
        result.update(aggregated_metrics)
    return EvaluationResult(**result)


def compute_metrics_from_file(src_mcap_path: PathLike, dst_mcap_path: PathLike) -> EvaluationResult:
    """Evaluate the accuracy of the agent's predictions."""
    with OWAMcapReader(src_mcap_path) as src_reader, OWAMcapReader(dst_mcap_path) as dst_reader:
        src_events = src_reader.iter_messages(topics=["screen", "keyboard", "mouse/raw"])
        dst_events = dst_reader.iter_messages(topics=["screen", "keyboard", "mouse/raw"])
        return compute_metrics(src_events, dst_events)


if __name__ == "__main__":
    result = compute_metrics_from_file("src.mcap", "dst.mcap")
    print(result)
