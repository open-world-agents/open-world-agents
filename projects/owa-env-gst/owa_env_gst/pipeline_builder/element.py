from abc import ABC, abstractmethod


class StringLike(ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass


StringLike.register(str)


class Element(StringLike):
    def __init__(self, factory_name: str, properties: dict = None):
        self.factory_name = factory_name
        self.properties = properties or {}

    def __repr__(self):
        return f"Element({self.factory_name}, {self.properties})"

    def __str__(self):
        properties_str = " ".join(f"{k}={v}" for k, v in self.properties.items())
        return f"{self.factory_name} {properties_str}".strip()

    def __hash__(self):
        return hash((self.factory_name, frozenset(self.properties.items())))

    def __eq__(self, other):
        return (
            isinstance(other, Element)
            and self.factory_name == other.factory_name
            and self.properties == other.properties
        )

    def __rshift__(self, other):
        return Pipeline(str(self)) >> other

    def __rrshift__(self, other):
        return other >> Pipeline(str(self))

    def __or__(self, other):
        return Pipeline(str(self)) | other

    def __ror__(self, other):
        return other | Pipeline(str(self))


class Pipeline(StringLike):
    """
    You can use the `|` operator to concatenate two elements and the `>>` operator to chain them.
    To concatenate n elements, you can use `functools.reduce` or `itertools.accumulate`.
    """

    def __init__(self, string: str):
        self.string = string

    def __str__(self):
        return self.string

    def __rshift__(self, other: StringLike):
        return Pipeline(f"{self} ! {other}")

    def __rrshift__(self, other: StringLike):
        return Pipeline(f"{other} ! {self}")

    def __or__(self, other: StringLike):
        if len(self.string) == 0:
            return Pipeline(str(other))
        return Pipeline(f"{self} {other}")

    def __ror__(self, other: StringLike):
        if len(self.string) == 0:
            return Pipeline(str(other))
        return Pipeline(f"{other} {self}")


if __name__ == "__main__":
    # Testing the provided examples
    muxer = Element("matroskamux", {"name": "mux"})
    a = Element("a")
    b = Element("b")
    c = Element("c")
    demuxer = Element("demuxer")

    # Example 1
    A = (a >> "mux." | b >> "mux." | c >> "mux.") | muxer
    B = "a ! mux. b ! mux. c ! mux. matroskamux name=mux"
    assert str(A) == B

    # Example 2
    A = demuxer | (a | b | c)
    assert str(A) == "demuxer a b c"

    print("All tests passed!")
