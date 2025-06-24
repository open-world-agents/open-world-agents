import pytest

from owa.data.interval.interval import Interval, Intervals


class TestInterval:
    """Test cases for the Interval class."""

    def test_interval_creation(self):
        """Test basic interval creation."""
        interval = Interval(start=1, end=5)
        assert interval.start == 1
        assert interval.end == 5

    def test_interval_validation(self):
        """Test interval validation."""
        with pytest.raises(ValueError, match="Invalid interval"):
            Interval(start=5, end=1)

        with pytest.raises(ValueError, match="Invalid interval"):
            Interval(start=3, end=3)

    def test_interval_contains(self):
        """Test interval containment."""
        interval = Interval(start=1, end=5)
        assert 1 in interval
        assert 3 in interval
        assert 4 in interval
        assert 5 not in interval  # Closed-open interval
        assert 0 not in interval

    def test_interval_properties(self):
        """Test interval properties."""
        interval = Interval(start=2, end=7)
        assert interval.length == 5
        assert repr(interval) == "[2, 7)"
        assert list(interval) == [2, 7]

    def test_interval_overlaps(self):
        """Test interval overlap detection."""
        interval1 = Interval(start=1, end=5)
        interval2 = Interval(start=3, end=7)
        interval3 = Interval(start=6, end=9)

        assert interval1.overlaps(interval2)
        assert interval2.overlaps(interval1)
        assert not interval1.overlaps(interval3)

    def test_interval_adjacent(self):
        """Test interval adjacency detection."""
        interval1 = Interval(start=1, end=5)
        interval2 = Interval(start=5, end=8)
        interval3 = Interval(start=6, end=9)

        assert interval1.adjacent_to(interval2)
        assert interval2.adjacent_to(interval1)
        assert not interval1.adjacent_to(interval3)


class TestIntervals:
    """Test cases for the Intervals class."""

    def test_empty_initialization(self):
        """Test empty intervals initialization."""
        intervals = Intervals()
        assert intervals.is_empty
        assert len(intervals) == 0
        assert intervals.total_length == 0
        assert list(intervals) == []

    def test_initialization_from_tuples(self):
        """Test initialization from tuples."""
        intervals = Intervals([(1, 3), (5, 7)])
        assert len(intervals) == 2
        assert intervals.to_tuples() == [(1, 3), (5, 7)]

    def test_initialization_from_intervals(self):
        """Test initialization from Interval objects."""
        interval_objs = [Interval(start=1, end=3), Interval(start=5, end=7)]
        intervals = Intervals(interval_objs)
        assert len(intervals) == 2
        assert intervals.to_tuples() == [(1, 3), (5, 7)]

    def test_initialization_with_overlapping_intervals(self):
        """Test that overlapping intervals are merged during initialization."""
        intervals = Intervals([(1, 4), (3, 6), (8, 10)])
        assert len(intervals) == 2
        assert intervals.to_tuples() == [(1, 6), (8, 10)]

    def test_from_range(self):
        """Test creating intervals from range."""
        intervals = Intervals.from_range(1, 5)
        assert len(intervals) == 1
        assert intervals.to_tuples() == [(1, 5)]

    def test_invalid_initialization(self):
        """Test invalid initialization raises errors."""
        with pytest.raises(TypeError):
            Intervals([1, 2, 3])  # Not tuples or Intervals

        with pytest.raises(TypeError):
            Intervals([(1,)])  # Invalid tuple length

    def test_contains(self):
        """Test value containment."""
        intervals = Intervals([(1, 3), (5, 7)])
        assert 1 in intervals
        assert 2 in intervals
        assert 3 not in intervals  # Closed-open
        assert 4 not in intervals
        assert 5 in intervals
        assert 6 in intervals
        assert 7 not in intervals

    def test_add_method(self):
        """Test adding intervals."""
        intervals = Intervals()
        intervals.add((1, 3))
        assert intervals.to_tuples() == [(1, 3)]

        intervals.add((5, 7))
        assert intervals.to_tuples() == [(1, 3), (5, 7)]

        # Test merging
        intervals.add((2, 6))
        assert intervals.to_tuples() == [(1, 7)]

    def test_add_method_chaining(self):
        """Test method chaining with add."""
        intervals = Intervals().add((1, 3)).add((5, 7)).add((2, 6))
        assert intervals.to_tuples() == [(1, 7)]

    def test_union_operation(self):
        """Test union operation."""
        intervals1 = Intervals([(1, 3), (5, 7)])
        intervals2 = Intervals([(2, 4), (8, 9)])

        result = intervals1.union(intervals2)
        assert result.to_tuples() == [(1, 4), (5, 7), (8, 9)]

        # Test operator version
        result = intervals1 | intervals2
        assert result.to_tuples() == [(1, 4), (5, 7), (8, 9)]

    def test_intersection_operation(self):
        """Test intersection operation."""
        intervals1 = Intervals([(1, 5), (7, 9)])
        intervals2 = Intervals([(2, 4), (8, 10)])

        result = intervals1.intersection(intervals2)
        assert result.to_tuples() == [(2, 4), (8, 9)]

        # Test operator version
        result = intervals1 & intervals2
        assert result.to_tuples() == [(2, 4), (8, 9)]

    def test_intersection_no_overlap(self):
        """Test intersection with no overlap."""
        intervals1 = Intervals([(1, 3), (5, 7)])
        intervals2 = Intervals([(4, 5), (8, 9)])

        result = intervals1.intersection(intervals2)
        assert result.is_empty

    def test_difference_operation(self):
        """Test difference operation."""
        intervals1 = Intervals([(1, 10)])
        intervals2 = Intervals([(3, 5), (7, 8)])

        result = intervals1.difference(intervals2)
        assert result.to_tuples() == [(1, 3), (5, 7), (8, 10)]

        # Test operator version
        result = intervals1 - intervals2
        assert result.to_tuples() == [(1, 3), (5, 7), (8, 10)]

    def test_difference_complete_removal(self):
        """Test difference that completely removes intervals."""
        intervals1 = Intervals([(1, 5), (7, 9)])
        intervals2 = Intervals([(0, 10)])

        result = intervals1.difference(intervals2)
        assert result.is_empty

    def test_difference_no_overlap(self):
        """Test difference with no overlap."""
        intervals1 = Intervals([(1, 3), (5, 7)])
        intervals2 = Intervals([(4, 5), (8, 9)])

        result = intervals1.difference(intervals2)
        assert result.to_tuples() == [(1, 3), (5, 7)]

    def test_difference_partial_overlap(self):
        """Test difference with partial overlaps."""
        intervals1 = Intervals([(1, 6)])
        intervals2 = Intervals([(3, 4)])

        result = intervals1.difference(intervals2)
        assert result.to_tuples() == [(1, 3), (4, 6)]

    def test_clear_method(self):
        """Test clearing all intervals."""
        intervals = Intervals([(1, 3), (5, 7)])
        assert not intervals.is_empty

        intervals.clear()
        assert intervals.is_empty
        assert len(intervals) == 0

    def test_properties(self):
        """Test various properties."""
        intervals = Intervals([(1, 3), (5, 9)])

        assert intervals.total_length == 6  # (3-1) + (9-5)
        assert not intervals.is_empty
        assert len(intervals) == 2

    def test_repr(self):
        """Test string representation."""
        empty_intervals = Intervals()
        assert repr(empty_intervals) == "Intervals()"

        intervals = Intervals([(1, 3), (5, 7)])
        assert "Intervals([" in repr(intervals)

    def test_normalization_sorting(self):
        """Test that intervals are properly sorted and normalized."""
        intervals = Intervals([(5, 7), (1, 3), (2, 4)])
        assert intervals.to_tuples() == [(1, 4), (5, 7)]

    def test_adjacent_interval_merging(self):
        """Test that adjacent intervals are merged."""
        intervals = Intervals([(1, 3), (3, 5)])
        assert intervals.to_tuples() == [(1, 5)]

    def test_complex_operations(self):
        """Test complex combinations of operations."""
        intervals1 = Intervals([(1, 10), (15, 20)])
        intervals2 = Intervals([(5, 8), (12, 17)])

        # Complex difference
        result = intervals1 - intervals2
        expected = [(1, 5), (8, 10), (17, 20)]
        assert result.to_tuples() == expected

        # Complex union
        result = intervals1 | intervals2
        expected = [(1, 10), (12, 20)]
        assert result.to_tuples() == expected

    def test_edge_cases(self):
        """Test edge cases."""
        # Single point intervals (should raise error)
        with pytest.raises(ValueError):
            Intervals([(1, 1)])

        # Empty intervals operations
        empty = Intervals()
        non_empty = Intervals([(1, 3)])

        assert (empty | non_empty).to_tuples() == [(1, 3)]
        assert (empty & non_empty).is_empty
        assert (empty - non_empty).is_empty
        assert (non_empty - empty).to_tuples() == [(1, 3)]

    def test_iteration(self):
        """Test iteration over intervals."""
        intervals = Intervals([(1, 3), (5, 7)])
        interval_list = list(intervals)

        assert len(interval_list) == 2
        assert interval_list[0].start == 1
        assert interval_list[0].end == 3
        assert interval_list[1].start == 5
        assert interval_list[1].end == 7


if __name__ == "__main__":
    pytest.main([__file__])
