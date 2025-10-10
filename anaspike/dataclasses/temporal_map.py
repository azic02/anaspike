from typing import Generic, TypeVar, Iterator

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
    def along_time_dim(self) -> Iterator[NDArray[ElmnT]]:
        yield from self.__elements

