from typing import Iterable, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ..dataclasses.interval import Bin
from ._helpers import offset_via_slicing



def pearson_correlation_offset_data(static_data: NDArray[np.float64], to_offset_data: NDArray[np.float64], offset_vectors: Iterable[NDArray[np.int64]]) -> NDArray[np.float64]:
    return np.array([np.corrcoef(*map(np.ravel, offset_via_slicing(static_data, to_offset_data, v)))[0,1] for v in offset_vectors])


def radial_average(origin: Tuple[float,float], x_pos: NDArray[np.float64], y_pos: NDArray[np.float64], data: NDArray[np.float64], radial_bins: Sequence[Bin]) -> NDArray[np.float64]:
    r = np.sqrt((x_pos - origin[0])**2 + (y_pos - origin[1])**2)
    bin_edges = np.array([b.start for b in radial_bins] + [radial_bins[-1].end])
    sums, _ = np.histogram(r, bins=bin_edges, weights=data)
    counts, _ = np.histogram(r, bins=bin_edges)
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)


def morans_i(data: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    if data.ndim != 1:
        raise ValueError("Data must be a 1D array")
    n = len(data)
    if weights.shape != (n, n):
        raise ValueError("Weights must be a square matrix of shape (n, n), where n is the length of data")
    w = np.sum(weights, dtype=np.float64)
    diff = data - np.mean(data)
    numerator = np.sum(weights * np.outer(diff, diff))
    denominator = np.sum(diff**2)
    return (n / w) * (numerator / denominator)

