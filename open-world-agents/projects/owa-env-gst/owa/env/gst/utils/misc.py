from fractions import Fraction


def framerate_float_to_str(fps: float) -> str:
    """
    Convert a float framerate to a string representation.

    Args:
        fps (float): The framerate as a float. e.g. 60.0

    Returns:
        str: The framerate as a string in the format "numerator/denominator". e.g. "60/1"
    """
    frac = Fraction(fps).limit_denominator()
    return f"{frac.numerator}/{frac.denominator}"
