from typing import Callable


class Pipe:
    """
    A class to create a pipeline of functions that can be executed in sequence.

    This class allows you to chain functions together, passing the output of one
    function as the input to the next.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the Pipe with optional arguments and keyword arguments.

        These arguments will be passed to the first function in the pipeline.
        """
        self._args = args
        self._kwargs = kwargs
        self._pipe = []

    def __or__(self, other: "Callable | Pipe") -> "Pipe":
        """
        Add a function or another Pipe to the pipeline using the '|' operator.

        Args:
            other (Callable | Pipe): A callable function or another Pipe instance.

        Returns:
            Pipe: The updated Pipe instance.

        Raises:
            TypeError: If the provided object is neither a callable nor a Pipe.
        """
        if isinstance(other, Pipe):
            self._pipe.extend(other._pipe)
        elif callable(other):
            self._pipe.append(other)
        else:
            raise TypeError(f"Unsupported type for pipe: {type(other)}")
        return self

    def execute(self):
        """
        Execute the functions in the pipeline in sequence.

        The result of each function is passed as the input to the next function.

        Returns:
            Any: The final result after executing all functions in the pipeline.

        Raises:
            ValueError: If the pipeline is empty and there are no functions to execute.
        """
        if not self._pipe:
            raise ValueError("No functions in the pipe to execute.")
        result = self._pipe[0](*self._args, **self._kwargs)
        for func in self._pipe[1:]:
            result = func(result)
        return result
