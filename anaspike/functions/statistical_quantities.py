from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import pearsonr

from ._helpers import offset_via_slicing



def pearson_correlation_offset_data(static_data: NDArray[np.float64], to_offset_data: NDArray[np.float64], offset_vectors: Iterable[NDArray[np.int64]]) -> NDArray[np.float64]:
    return np.array([pearsonr(*map(np.ravel, offset_via_slicing(static_data, to_offset_data, v))).statistic for v in offset_vectors])
