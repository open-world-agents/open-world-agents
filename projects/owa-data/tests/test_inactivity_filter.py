import tempfile
from pathlib import Path

import pytest

from owa.core.time import TimeUnits
from owa.data.interval.selector import InactivityFilter


class MockMessage:
    """Mock MCAP message for testing."""

    def __init__(self, topic: str, timestamp: int, decoded=None):
        self.topic = topic
        self.timestamp = timestamp
        self.decoded = decoded or {}


class MockReader:
    """Mock MCAP reader that supports context manager protocol."""

    def __init__(self, messages):
        self.messages = messages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def iter_messages(self, topics=None, reverse=False):
        messages = self.messages
        if topics is not None:
            messages = [msg for msg in messages if msg.topic in topics]

        if reverse:
            messages = reversed(messages)

        yield from messages


def create_test_mcap_file(messages: list[MockMessage]) -> Path:
    """Create a temporary MCAP file with test messages."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".mcap", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    # For testing, we'll mock the OWAMcapReader to return our test messages
    return temp_path


class TestInactivityFilter:
    """Test cases for the enhanced InactivityFilter class."""

    def test_initialization_default_thresholds(self):
        """Test InactivityFilter initialization with default thresholds."""
        filter_obj = InactivityFilter()
        assert filter_obj.screen_inactivity_threshold == 1.0
        assert filter_obj.input_inactivity_threshold == 5.0

    def test_initialization_custom_thresholds(self):
        """Test InactivityFilter initialization with custom thresholds."""
        filter_obj = InactivityFilter(screen_inactivity_threshold=2.0, input_inactivity_threshold=10.0)
        assert filter_obj.screen_inactivity_threshold == 2.0
        assert filter_obj.input_inactivity_threshold == 10.0

    def test_no_screen_events(self, monkeypatch):
        """Test behavior when no screen events are present."""
        messages = [
            MockMessage("keyboard", int(1.0 * TimeUnits.SECOND)),
            MockMessage("mouse/raw", int(2.0 * TimeUnits.SECOND)),
        ]

        def mock_reader_constructor(path):
            return MockReader(messages)

        monkeypatch.setattr("owa.data.interval.selector.OWAMcapReader", mock_reader_constructor)

        filter_obj = InactivityFilter()
        temp_path = create_test_mcap_file(messages)

        result = filter_obj.extract_intervals(temp_path)
        assert result.is_empty

    def test_screen_boundary_detection(self, monkeypatch):
        """Test screen boundary interval detection."""
        messages = [
            MockMessage("screen", int(1.0 * TimeUnits.SECOND)),
            MockMessage("keyboard", int(1.5 * TimeUnits.SECOND)),
            MockMessage("screen", int(2.0 * TimeUnits.SECOND)),
            MockMessage("screen", int(5.0 * TimeUnits.SECOND)),
        ]

        def mock_reader_constructor(path):
            return MockReader(messages)

        monkeypatch.setattr("owa.data.interval.selector.OWAMcapReader", mock_reader_constructor)

        filter_obj = InactivityFilter()
        temp_path = create_test_mcap_file(messages)

        boundary = filter_obj._get_screen_boundary_interval(temp_path)
        assert not boundary.is_empty
        assert boundary.to_tuples() == [(int(1.0 * TimeUnits.SECOND), int(5.0 * TimeUnits.SECOND))]

    def test_screen_activity_with_small_gaps(self, monkeypatch):
        """Test screen activity intervals with gaps smaller than threshold."""
        messages = [
            MockMessage("screen", int(1.0 * TimeUnits.SECOND)),
            MockMessage("screen", int(1.5 * TimeUnits.SECOND)),  # 0.5s gap - should be continuous
            MockMessage("screen", int(2.0 * TimeUnits.SECOND)),  # 0.5s gap - should be continuous
        ]

        def mock_reader_constructor(path):
            return MockReader(messages)

        monkeypatch.setattr("owa.data.interval.selector.OWAMcapReader", mock_reader_constructor)

        filter_obj = InactivityFilter()
        temp_path = create_test_mcap_file(messages)

        activity = filter_obj._get_topic_activity_intervals(temp_path, ["screen"], 1.0)
        assert not activity.is_empty
        # Should be one continuous interval
        assert len(activity) == 1
        assert activity.to_tuples() == [(int(1.0 * TimeUnits.SECOND), int(2.0 * TimeUnits.SECOND))]

    def test_screen_activity_with_large_gaps(self, monkeypatch):
        """Test screen activity intervals with gaps larger than threshold."""
        messages = [
            MockMessage("screen", int(1.0 * TimeUnits.SECOND)),
            MockMessage("screen", int(1.5 * TimeUnits.SECOND)),  # 0.5s gap - continuous
            MockMessage("screen", int(4.0 * TimeUnits.SECOND)),  # 2.5s gap - should break interval
            MockMessage("screen", int(4.5 * TimeUnits.SECOND)),  # 0.5s gap - continuous
        ]

        def mock_reader_constructor(path):
            return MockReader(messages)

        monkeypatch.setattr("owa.data.interval.selector.OWAMcapReader", mock_reader_constructor)

        filter_obj = InactivityFilter()
        temp_path = create_test_mcap_file(messages)

        activity = filter_obj._get_topic_activity_intervals(temp_path, ["screen"], 1.0)
        assert not activity.is_empty
        # Should be two separate intervals
        assert len(activity) == 2
        expected = [
            (int(1.0 * TimeUnits.SECOND), int(1.5 * TimeUnits.SECOND)),
            (int(4.0 * TimeUnits.SECOND), int(4.5 * TimeUnits.SECOND)),
        ]
        assert activity.to_tuples() == expected

    def test_input_device_activity_with_different_threshold(self, monkeypatch):
        """Test input device activity intervals with 5-second threshold."""
        messages = [
            MockMessage("keyboard", int(1.0 * TimeUnits.SECOND)),
            MockMessage("mouse/raw", int(3.0 * TimeUnits.SECOND)),  # 2s gap - should be continuous
            MockMessage("keyboard", int(4.0 * TimeUnits.SECOND)),  # 1s gap - should be continuous
            MockMessage("mouse/raw", int(10.0 * TimeUnits.SECOND)),  # 6s gap - should break interval
            MockMessage("keyboard", int(11.0 * TimeUnits.SECOND)),  # 1s gap - continuous
        ]

        def mock_reader_constructor(path):
            return MockReader(messages)

        monkeypatch.setattr("owa.data.interval.selector.OWAMcapReader", mock_reader_constructor)

        filter_obj = InactivityFilter()
        temp_path = create_test_mcap_file(messages)

        activity = filter_obj._get_topic_activity_intervals(temp_path, ["keyboard", "mouse/raw"], 5.0)
        assert not activity.is_empty
        # Should be two separate intervals due to 6s gap
        assert len(activity) == 2
        expected = [
            (int(1.0 * TimeUnits.SECOND), int(4.0 * TimeUnits.SECOND)),
            (int(10.0 * TimeUnits.SECOND), int(11.0 * TimeUnits.SECOND)),
        ]
        assert activity.to_tuples() == expected

    def test_composite_interval_operations(self, monkeypatch):
        """Test the full composite interval operations in extract_intervals."""
        # Create a scenario with screen events and input events
        messages = [
            # Screen events define boundary: 1-6 seconds
            MockMessage("screen", int(1.0 * TimeUnits.SECOND)),
            MockMessage("screen", int(2.0 * TimeUnits.SECOND)),
            MockMessage("screen", int(6.0 * TimeUnits.SECOND)),  # Last screen event
            # Input events with some gaps
            MockMessage("keyboard", int(1.5 * TimeUnits.SECOND)),
            MockMessage("mouse/raw", int(2.5 * TimeUnits.SECOND)),
            MockMessage("keyboard", int(5.5 * TimeUnits.SECOND)),
        ]

        def mock_reader_constructor(path):
            return MockReader(messages)

        monkeypatch.setattr("owa.data.interval.selector.OWAMcapReader", mock_reader_constructor)

        filter_obj = InactivityFilter()
        temp_path = create_test_mcap_file(messages)

        result = filter_obj.extract_intervals(temp_path)
        assert not result.is_empty

        # The result should be the intersection of screen activity, input activity,
        # and screen boundary
        # Screen boundary: [1.0, 6.0]
        # Screen activity: [1.0, 2.0] (gap from 2.0 to 6.0 = 4s > 1s threshold, so interval breaks; single-event interval at 6.0 is ignored)
        # Input activity: [1.5, 5.5] (no gaps > 5s)
        # Intersection: [1.0, 2.0] ∩ [1.5, 5.5] = [1.5, 2.0]
        #               [6.0, 6.0] ∩ [1.5, 5.5] = empty (6.0 > 5.5)
        # Final result should be [1.5, 2.0]
        expected_start = int(1.5 * TimeUnits.SECOND)
        expected_end = int(2.0 * TimeUnits.SECOND)
        assert result.to_tuples() == [(expected_start, expected_end)]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
