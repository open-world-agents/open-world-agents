from owa.registry import CALLABLES


@CALLABLES.register("example/callable")
class ExampleCallable:
    """
    Callable must implement the __call__ method to be callable.
    You can add more functionality inside __call__ as needed.
    """

    def __call__(self):
        # Replace the following line with your actual implementation.
        pass


@CALLABLES.register("example/print")
def example_print():
    print("Hello, World!")
    return "Hello, World!"
