from owa.registry import CALLABLES


@CALLABLES.register("example/callable")
class ExampleCallable:
    def __call__(self):
        # Implement here!
        pass


@CALLABLES.register("example/print")
def example_print():
    print("Hello, World!")
    return "Hello, World!"
