from __future__ import annotations
import numpy as np
import typing

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray
    from collections.abc import Callable


def convert_to_int(value: int | float | np.number[typing.Any]) -> int:
    if int(value) == float(value):
        return int(value)
    raise ValueError(f"Value '{value}' cannot be converted to an integer")


def conditional_replace(
    array: NDArray[typing.Any],
    replacement: int | float | np.number[typing.Any],
    condition_func: Callable[[NDArray[typing.Any]], NDArray[np.bool_]],
) -> None:
    array[condition_func(array)] = replacement
