from typing import Any, Sized, Iterable, Tuple

import numpy as np
from numpy.typing import NDArray



def slice_from_vec(v: Iterable[int]) -> Tuple[slice, ...]:
    return tuple(slice(x, None) if x >= 0 else slice(None, x) for x in v)


def offset_via_slicing(static_data: NDArray, to_offset_data: NDArray, offset_vector: NDArray[np.int64]) -> Tuple[NDArray, NDArray]:
    if static_data.shape != to_offset_data.shape:
        raise ValueError("The two datasets must have the same shape.")
    if offset_vector.shape[0] != static_data.ndim:
        raise ValueError("The offset vector must have the same number of dimensions as the datasets.")
    return static_data[slice_from_vec(offset_vector)], to_offset_data[slice_from_vec(-offset_vector)]


def validate_same_length(*args: Sized) -> None:
    def __are_equal(c: Iterable[Any]) -> bool:
        return len(set(c)) <= 1

    if not args:
        return
    lengths = (len(it) for it in args)
    if not __are_equal(lengths):
        raise ValueError("Not all given sized objects have same length.")


def validate_one_dimensional(arr: NDArray[Any]):
    if arr.ndim != 1:
        raise ValueError("Array not one dimensional.")

