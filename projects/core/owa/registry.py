# ================ Definition of the Registry class ================================
# references:
# - https://github.com/open-mmlab/mmdetection/blob/main/mmdet/registry.py
# - https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html


class Registry:
    def __init__(self):
        self._registry = {}

    def register(self, name: str):
        def decorator(cls):
            self._registry[name] = cls
            return cls

        return decorator

    def __getitem__(self, name: str):
        return self._registry[name]

    # list all the registered items
    def __repr__(self):
        return self._registry.__repr__()


CALLABLES = Registry()
LISTENERS = Registry()
