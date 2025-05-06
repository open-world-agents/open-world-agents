from dataclasses import dataclass

import numpy as np

from mcap_owa.highlevel import OWAMcapReader
from owa.core.time import TimeUnits

from ..clock import Clock
from ..event import Event
from .agent import Agent


@dataclass
class DTStrategy:
    start: float
    end: float
    delta: float  # if delta is 0, it means the event type is discrete event.
    topic: str


DELTA_TIMESTAMPS = {
    "screen": DTStrategy(-0.25, 0, 0.05, "screen"),
    "keyboard": DTStrategy(-0.25, 0, 0, "keyboard"),
    "mouse": DTStrategy(-0.25, 0, 0.05, "mouse"),
    "keyboard_label": DTStrategy(0, 0.25, 0, "keyboard"),
    "mouse_label": DTStrategy(0, 0.25, 0.05, "mouse"),
}


def sample_interval():
    return np.random.rand() * (0.25 + 0.25)  # past 0.25 + future 0.25


class OfflineAgentExecutor:
    def __init__(self, clock: Clock): ...
    def register(self, agent: Agent):
        self.agent = agent

    def add_event(self, event):
        for agent in self.agents:
            agent.event_queue.append(event)

    def consume(self, mcap_file: str):
        with OWAMcapReader(mcap_file) as reader:
            anchor = reader.start_time
            next_time = anchor + TimeUnits.SECOND * sample_interval()
            for topic, timestamp, msg in reader.iter_decoded_messages():
                self.add_event(Event(topic, timestamp, msg))
                if timestamp >= next_time:
                    self.agent.prepare_input()
                    anchor = timestamp
                    next_time = anchor + TimeUnits.SECOND * sample_interval()
