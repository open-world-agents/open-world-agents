from typing import Any, Iterator, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, model_validator


class Interval(BaseModel):
    """
    Represents a closed-open interval [start, end).
    """

    start: int
    end: int

    @model_validator(mode="after")
    def validate_interval(self) -> "Interval":
        if self.start >= self.end:
            raise ValueError(f"Invalid interval: start ({self.start}) must be less than end ({self.end})")
        return self

    def __contains__(self, value: int) -> bool:
        """Check if a value is contained in this interval."""
        return self.start <= value < self.end

    def __repr__(self) -> str:
        return f"[{self.start}, {self.end})"

    def __iter__(self) -> Iterator[int]:
        yield self.start
        yield self.end

    @property
    def length(self) -> int:
        """Return the length of this interval."""
        return self.end - self.start

    def overlaps(self, other: "Interval") -> bool:
        """Check if this interval overlaps with another interval."""
        return max(self.start, other.start) < min(self.end, other.end)

    def adjacent_to(self, other: "Interval") -> bool:
        """Check if this interval is adjacent to another interval."""
        return self.end == other.start or other.end == self.start


class Intervals(BaseModel):
    """
    Represents a collection of non-overlapping intervals.
    All operations automatically normalize (merge overlapping intervals).
    """

    intervals: List[Interval] = Field(default_factory=list)

    def __init__(self, intervals: Optional[List[Union[Interval, Tuple[int, int]]]] = None, **kwargs: Any):
        """
        Initialize with a list of Interval objects or (start, end) tuples.

        Args:
            intervals: List of Interval objects or (start, end) tuples

        Examples:
            # Create from tuples
            intervals = Intervals([(1, 3), (5, 7)])

            # Create from Interval objects
            intervals = Intervals([Interval(start=1, end=3), Interval(start=5, end=7)])

            # Empty intervals
            intervals = Intervals()
        """
        if intervals is None:
            intervals = []

        parsed_intervals = []
        for interval in intervals:
            if isinstance(interval, tuple) and len(interval) == 2:
                parsed_intervals.append(Interval(start=interval[0], end=interval[1]))
            elif isinstance(interval, Interval):
                parsed_intervals.append(interval)
            else:
                raise TypeError(f"Expected Interval or (start, end) tuple, got {type(interval)}")

        super().__init__(intervals=parsed_intervals, **kwargs)
        self._normalize()

    @classmethod
    def from_range(cls, start: int, end: int) -> "Intervals":
        """Create Intervals containing a single interval from start to end."""
        return cls([(start, end)])

    def to_tuples(self) -> List[Tuple[int, int]]:
        """Convert to a list of (start, end) tuples."""
        return [(interval.start, interval.end) for interval in self.intervals]

    def __repr__(self) -> str:
        if not self.intervals:
            return "Intervals()"
        return f"Intervals({self.intervals})"

    def __iter__(self) -> Iterator[Interval]:
        """Iterate through the intervals."""
        return iter(self.intervals)

    def __contains__(self, value: int) -> bool:
        """Check if a value is contained in any interval."""
        return any(value in interval for interval in self.intervals)

    def __len__(self) -> int:
        """Return the number of intervals."""
        return len(self.intervals)

    @property
    def is_empty(self) -> bool:
        """Check if there are no intervals."""
        return len(self.intervals) == 0

    @property
    def total_length(self) -> int:
        """Return the sum of all interval lengths."""
        return sum(interval.length for interval in self.intervals)

    def _normalize(self) -> None:
        """Merge overlapping intervals and sort them."""
        if not self.intervals:
            return

        self.intervals.sort(key=lambda x: x.start)

        i = 0
        while i < len(self.intervals) - 1:
            current = self.intervals[i]
            next_interval = self.intervals[i + 1]

            # If intervals overlap or are adjacent, merge them
            if current.end >= next_interval.start:
                current.end = max(current.end, next_interval.end)
                self.intervals.pop(i + 1)
            else:
                i += 1

    def add(self, interval: Union[Interval, Tuple[int, int]]) -> "Intervals":
        """
        Add an interval and normalize.

        Args:
            interval: Interval object or (start, end) tuple

        Returns:
            Self for method chaining

        Example:
            intervals = Intervals()
            intervals.add((1, 3)).add((5, 7))
        """
        if isinstance(interval, tuple) and len(interval) == 2:
            interval = Interval(start=interval[0], end=interval[1])
        elif not isinstance(interval, Interval):
            raise TypeError(f"Expected Interval or (start, end) tuple, got {type(interval)}")

        self.intervals.append(interval)
        self._normalize()
        return self

    def union(self, other: "Intervals") -> "Intervals":
        """
        Return the union of this Intervals with another.

        Example:
            intervals1 = Intervals([(1, 3), (5, 7)])
            intervals2 = Intervals([(2, 4), (8, 9)])
            result = intervals1.union(intervals2)  # [(1, 4), (5, 7), (8, 9)]
        """
        result = Intervals()
        result.intervals = self.intervals.copy()
        for interval in other.intervals:
            result.intervals.append(interval)
        result._normalize()
        return result

    def __or__(self, other: "Intervals") -> "Intervals":
        """Operator version of union: intervals1 | intervals2"""
        return self.union(other)

    def intersection(self, other: "Intervals") -> "Intervals":
        """
        Return the intersection of this Intervals with another.

        Example:
            intervals1 = Intervals([(1, 5), (7, 10)])
            intervals2 = Intervals([(2, 4), (8, 10)])
            result = intervals1.intersection(intervals2)  # [(2, 4), (8, 9)]
        """
        result = Intervals()

        for interval1 in self.intervals:
            for interval2 in other.intervals:
                if interval1.overlaps(interval2):
                    result.add((max(interval1.start, interval2.start), min(interval1.end, interval2.end)))

        return result

    def __and__(self, other: "Intervals") -> "Intervals":
        """Operator version of intersection: intervals1 & intervals2"""
        return self.intersection(other)

    def difference(self, other: "Intervals") -> "Intervals":
        """
        Return the difference: intervals in self that are not in other.

        Example:
            intervals1 = Intervals([(1, 10)])
            intervals2 = Intervals([(3, 5), (7, 8)])
            result = intervals1.difference(intervals2)  # [(1, 3), (5, 7), (8, 10)]
        """
        result = Intervals(self.to_tuples())  # Copy of self

        for interval in other.intervals:
            new_intervals = []

            for existing in result.intervals:
                # No overlap, keep existing interval unchanged
                if not existing.overlaps(interval):
                    new_intervals.append(existing)
                    continue

                # Interval completely contains existing, remove existing
                if interval.start <= existing.start and interval.end >= existing.end:
                    continue

                # Existing completely contains interval, split existing
                if existing.start < interval.start and existing.end > interval.end:
                    new_intervals.append(Interval(start=existing.start, end=interval.start))
                    new_intervals.append(Interval(start=interval.end, end=existing.end))
                    continue

                # Interval overlaps start of existing
                if interval.start <= existing.start and interval.end > existing.start and interval.end < existing.end:
                    new_intervals.append(Interval(start=interval.end, end=existing.end))
                    continue

                # Interval overlaps end of existing
                if interval.start > existing.start and interval.start < existing.end and interval.end >= existing.end:
                    new_intervals.append(Interval(start=existing.start, end=interval.start))
                    continue

            result.intervals = new_intervals

        result._normalize()
        return result

    def __sub__(self, other: "Intervals") -> "Intervals":
        """Operator version of difference: intervals1 - intervals2"""
        return self.difference(other)

    def clear(self) -> "Intervals":
        """Remove all intervals."""
        self.intervals.clear()
        return self


if __name__ == "__main__":
    # Example usage
    intervals = Intervals([(1, 3), (2, 4)])
    print(f"Original: {intervals}")  # Original: Intervals([1, 4))

    # Difference
    result = intervals - Intervals([(2, 3)])
    print(f"After subtraction: {result}")  # After subtraction: Intervals([1, 2), [3, 4))

    # Union
    result = Intervals([(1, 5)]) | Intervals([(7, 10)])
    print(f"Union example: {result}")  # Union example: Intervals([1, 5), [7, 10))

    # Intersection
    result = Intervals([(1, 5), (7, 10)]) & Intervals([(3, 8)])
    print(f"Intersection example: {result}")  # Intersection example: Intervals([3, 5), [7, 8))

    # Method chaining
    result = Intervals().add((1, 3)).add((5, 7)).add((2, 6))
    print(f"Chained additions: {result}")  # Chained additions: Intervals([1, 7))
