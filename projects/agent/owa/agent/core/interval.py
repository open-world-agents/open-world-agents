from typing import Any, List, Tuple

from pydantic import BaseModel, Field, model_validator


class IntervalUnit(BaseModel):
    """
    Represents a closed-open interval [start_time, end_time).
    """

    start_time: int
    end_time: int

    @model_validator(mode="after")
    def check_start_end(self) -> "IntervalUnit":
        assert self.start_time < self.end_time, (
            f"start_time {self.start_time} should be less than end_time {self.end_time}"
        )
        return self

    def __contains__(self, item: int) -> bool:
        return self.start_time <= item < self.end_time

    def __repr__(self) -> str:
        return f"({self.start_time}, {self.end_time})"

    def __or__(self, other: "IntervalUnit") -> List["IntervalUnit"]:
        # No overlap
        if self.end_time < other.start_time or other.end_time < self.start_time:
            return [self, other]
        # Overlap or touching
        return [
            IntervalUnit(
                start_time=min(self.start_time, other.start_time), end_time=max(self.end_time, other.end_time)
            )
        ]

    def __and__(self, other: "IntervalUnit") -> List["IntervalUnit"]:
        # No overlap
        if self.end_time <= other.start_time or other.end_time <= self.start_time:
            return []
        # Overlap
        return [
            IntervalUnit(
                start_time=max(self.start_time, other.start_time), end_time=min(self.end_time, other.end_time)
            )
        ]

    def __sub__(self, other: "IntervalUnit") -> List["IntervalUnit"]:
        # No overlap
        if self.end_time <= other.start_time or other.end_time <= self.start_time:
            return [self]
        # `other` completely covers `self`
        if other.start_time <= self.start_time and other.end_time >= self.end_time:
            return []
        # `other` is in the middle of `self`
        if other.start_time > self.start_time and other.end_time < self.end_time:
            return [
                IntervalUnit(start_time=self.start_time, end_time=other.start_time),
                IntervalUnit(start_time=other.end_time, end_time=self.end_time),
            ]
        # Overlapping at the start
        if other.start_time <= self.start_time:
            return [IntervalUnit(start_time=other.end_time, end_time=self.end_time)]
        # Overlapping at the end
        return [IntervalUnit(start_time=self.start_time, end_time=other.start_time)]


class IntervalUnion(BaseModel):
    """
    Represents a union of IntervalUnit objects.
    """

    intervals: List[IntervalUnit] = Field(default_factory=list)

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._normalize()

    @classmethod
    def from_tuple(cls, intervals: List[Tuple[int, int]]) -> "IntervalUnion":
        return cls(intervals=[IntervalUnit(start_time=interval[0], end_time=interval[1]) for interval in intervals])

    @classmethod
    def from_range(cls, start_time: int, end_time: int) -> "IntervalUnion":
        return cls(intervals=[IntervalUnit(start_time=start_time, end_time=end_time)])

    def to_tuple(self) -> List[Tuple[int, int]]:
        return [(interval.start_time, interval.end_time) for interval in self.intervals]

    def __contains__(self, item: int) -> bool:
        return any(item in interval for interval in self.intervals)

    @property
    def time_length(self) -> int:
        return sum(interval.end_time - interval.start_time for interval in self.intervals)

    def __repr__(self) -> str:
        return f"{self.intervals}"

    def _normalize(self):
        self.intervals = self._merge_intervals(sorted(self.intervals, key=lambda x: x.start_time))

    @staticmethod
    def _merge_intervals(intervals: List[IntervalUnit]) -> List[IntervalUnit]:
        if not intervals:
            return []
        merged = [intervals[0]]
        for current in intervals[1:]:
            last = merged[-1]
            if current.start_time <= last.end_time:
                last.end_time = max(last.end_time, current.end_time)
            else:
                merged.append(current)
        return merged

    def __or__(self, other: "IntervalUnion") -> "IntervalUnion":
        return IntervalUnion(intervals=self.intervals + other.intervals)

    def __and__(self, other: "IntervalUnion") -> "IntervalUnion":
        result_intervals = []
        for interval in self.intervals:
            for other_interval in other.intervals:
                result_intervals.extend(interval & other_interval)
        return IntervalUnion(intervals=result_intervals)

    def __sub__(self, other: "IntervalUnion") -> "IntervalUnion":
        result_intervals = self.intervals
        for other_interval in other.intervals:
            temp_result = []
            for interval in result_intervals:
                temp_result.extend(interval - other_interval)
            result_intervals = temp_result
        return IntervalUnion(intervals=result_intervals)

    def add(self, interval: IntervalUnit):
        self.intervals.append(interval)
        self._normalize()

    def clear(self):
        self.intervals.clear()


if __name__ == "__main__":
    interval = IntervalUnion(
        intervals=[IntervalUnit(start_time=1, end_time=3), IntervalUnit(start_time=2, end_time=4)]
    )
    print(interval)
    interval -= IntervalUnion.from_tuple([(2, 3)])
    print(interval)
