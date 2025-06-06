"""
Example callable components for the example environment plugin.

This module demonstrates how to create callable components using both
class-based and function-based approaches.
"""

from owa.core.registry import CALLABLES


@CALLABLES.register("example/callable")
class ExampleCallable:
    """Example callable class that can be called like a function."""

    def __call__(self) -> str:
        """
        Example callable implementation.

        Returns:
            A simple greeting message
        """
        return "Hello from ExampleCallable!"


@CALLABLES.register("example/print")
def example_print(message: str = "Hello, World!") -> str:
    """
    Example callable function that prints and returns a message.

    Args:
        message: The message to print and return

    Returns:
        The message that was printed
    """
    print(f"[Example Plugin] {message}")
    return message


@CALLABLES.register("example/add")
def example_add(a: int, b: int) -> int:
    """
    Example callable function that adds two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b

    Raises:
        TypeError: If arguments are not numbers
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Both arguments must be numbers")
    return a + b
