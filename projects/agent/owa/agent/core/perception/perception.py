import queue
from collections import defaultdict

from ..event import Event
from ..utils import iter_queue


class Perception(defaultdict[str, list[Event]]):
    """
    Perception is a defaultdict that maps topic names to lists of Event objects.
    Each Event object contains a timestamp, topic, and message.
    """

    def __init__(self):
        super().__init__(list)

    def __add__(self, other: "Perception") -> "Perception":
        """
        Merges two Perception objects by combining their events.
        """
        if not isinstance(other, Perception):
            raise TypeError("Can only merge with another Perception object.")
        merged = Perception()
        for topic in set(self.keys()) | set(other.keys()):
            merged[topic] = self.get(topic, []) + other.get(topic, [])
        return merged


class PerceptionQueue(defaultdict[str, queue.Queue]):
    def __init__(self):
        super().__init__(queue.Queue)

    def iter_queue(self) -> Perception:
        """
        Iterates over the queues in the PerceptionQueue and returns a Perception object.
        """
        perception = Perception()
        for topic, q in self.items():
            events = list(iter_queue(q))
            if events:
                perception[topic] = events
        return perception
