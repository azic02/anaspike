from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray



ValT = TypeVar("ValT", bound=np.generic)
class TemporalMap(Generic[ValT]):
    def __init__(self, times: NDArray[np.float64], values: NDArray[ValT]):
        if times.ndim != 1:
            raise ValueError("times must be a one-dimensional array.")
        if values.ndim < 1:
            raise ValueError("values must at least be a one-dimensional array.")
        if len(times) != values.shape[0]:
            raise ValueError("length of times must match the first dimension of values.")
       
        self.__times = times
        self.__values = values

    @property
    def ndim(self) -> int:
        return self.__values.ndim - 1

    @property
    def shape(self) -> tuple[int, ...]:
        return self.__values.shape[1:] if self.__values.ndim > 1 else ()

    @property
    def n_times(self) -> int:
        return len(self.__times)

    @property
    def times(self) -> NDArray[np.float64]:
        return self.__times

    @property
    def n_values(self) -> int:
        return self.__values.shape[1]

    @property
    def values(self) -> NDArray[ValT]:
        return self.__values

