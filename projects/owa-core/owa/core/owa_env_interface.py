from abc import ABC, abstractmethod
from types import ModuleType


class OwaEnvInterface(ModuleType, ABC):
    @abstractmethod
    def activate(self): ...
