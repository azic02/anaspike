from typing import Iterable, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import pearsonr

from ..dataclasses.interval import Bin
from ._helpers import offset_via_slicing



def pearson_correlation_offset_data(static_data: NDArray[np.float64], to_offset_data: NDArray[np.float64], offset_vectors: Iterable[NDArray[np.int64]]) -> NDArray[np.float64]:
    return np.array([pearsonr(*map(np.ravel, offset_via_slicing(static_data, to_offset_data, v))).statistic for v in offset_vectors])


def radial_average(origin: Tuple[float,float], x_pos: NDArray[np.float64], y_pos: NDArray[np.float64], data: NDArray[np.float64], radial_bins: Sequence[Bin]) -> NDArray[np.float64]:
    r = np.sqrt((x_pos - origin[0])**2 + (y_pos - origin[1])**2)
    bin_edges = np.array([b.start for b in radial_bins] + [radial_bins[-1].end])
    sums, _ = np.histogram(r, bins=bin_edges, weights=data)
    counts, _ = np.histogram(r, bins=bin_edges)
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)

