from numbers import Number
from typing import Callable
from numpy import ndarray


def convert_to_int(value: Number) -> int:
    if int(value) == float(value):
        return int(value)
    raise ValueError(f"Value '{value}' cannot be converted to an integer")


def conditional_replace(
    array: ndarray, replacement: Number, condition_func: Callable
) -> ndarray:
    array[condition_func(array)] = replacement
