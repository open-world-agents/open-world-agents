# shared/models.py
"""
Common data models shared between the evaluator and agent.

These models can be extended as needed to support more complex interactions.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


@dataclass
class TaskStep:
    """A single step within a task."""

    action: str
    params: Dict[str, Any]
    expected_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"action": self.action, "params": self.params}
        if self.expected_result:
            result["expected_result"] = self.expected_result
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskStep":
        return cls(action=data["action"], params=data["params"], expected_result=data.get("expected_result"))


@dataclass
class MetricValue:
    """A value for a particular metric."""

    name: str
    value: Union[float, int, str, bool]
    unit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"name": self.name, "value": self.value}
        if self.unit:
            result["unit"] = self.unit
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricValue":
        return cls(name=data["name"], value=data["value"], unit=data.get("unit"))


@dataclass
class LogEntry:
    """A log entry from task execution."""

    timestamp: datetime
    level: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {"timestamp": self.timestamp.isoformat(), "level": self.level, "message": self.message}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogEntry":
        return cls(timestamp=datetime.fromisoformat(data["timestamp"]), level=data["level"], message=data["message"])
