from typing import Any, Sized, Iterable, Tuple
from itertools import product

import numpy as np
from numpy.typing import NDArray



def construct_offsets(n: int, margin: int) -> NDArray[np.int64]:
    if n < margin:
        raise ValueError(f"n must be at least {margin}.")
    max_offset = n - margin
    return np.arange(-max_offset, max_offset + 1)


def construct_offset_vectors(n_x: int, n_y: int, margin: int) -> NDArray[np.int64]:
    x_offsets, y_offsets = construct_offsets(n_x, margin), construct_offsets(n_y, margin)
    return np.array(tuple(product(x_offsets, y_offsets)), dtype=np.int64)


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


def calculate_pairwise_2d_euclidean_distances(xs: NDArray, ys: NDArray) -> NDArray[np.float64]:
    validate_one_dimensional(xs)
    if xs.shape != ys.shape:
        raise ValueError("The two arrays must have the same shape.")
    return np.sqrt((xs[:, np.newaxis] - xs[np.newaxis, :])**2 + (ys[:, np.newaxis] - ys[np.newaxis, :])**2)

