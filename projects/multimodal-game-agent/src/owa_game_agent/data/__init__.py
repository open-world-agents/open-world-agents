from .query import OWAMcapQuery
from .sample import OWATrainingSample


class TimeUnits:
    NSECOND = 1
    USECOND = 10**3
    MSECOND = 10**6
    SECOND = 10**9


__all__ = ["OWAMcapQuery", "OWATrainingSample"]
