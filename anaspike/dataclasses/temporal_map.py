from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray



ElmnT = TypeVar("ElmnT", bound=np.generic)
class TemporalMap(Generic[ElmnT]):
    def __init__(self, times: NDArray[np.float64], elements: NDArray[ElmnT]):
        if times.ndim != 1:
            raise ValueError("times must be a one-dimensional array.")
        if elements.ndim < 1:
            raise ValueError("elements must at least be a one-dimensional array.")
        if len(times) != elements.shape[0]:
            raise ValueError("length of times must match the first dimension of elements.")
       
        self.__times = times
        self.__elements = elements

    @property
    def ndim(self) -> int:
        return self.__elements.ndim - 1

    @property
    def shape(self) -> tuple[int, ...]:
        return self.__elements.shape[1:] if self.__elements.ndim > 1 else ()

    @property
    def n_times(self) -> int:
        return len(self.__times)

    @property
    def times(self) -> NDArray[np.float64]:
        return self.__times

    @property
    def along_time_dim(self) -> NDArray[ElmnT]:
        return self.__elements


def correlation(tm1: TemporalMap[np.float64], tm2: TemporalMap[np.float64]) -> float:
    if tm1.ndim != 0 or tm2.ndim != 0:
        raise ValueError("Both TemporalMaps must be 0D (i.e., time series of scalars).")
    if tm1.n_times != tm2.n_times:
        raise ValueError("Both TemporalMaps must have the same number of time points.")
    return np.corrcoef(tm1.along_time_dim, tm2.along_time_dim)[0, 1]

