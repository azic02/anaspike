from typing import Optional

import numpy as np
from numpy.typing import NDArray



class Interval:
    def __init__(self, start: float, end: float):
        if start > end:
            raise ValueError("start must be less than or equal to end")
        self.__start = start
        self.__end = end

    @property
    def start(self) -> float:
        return self.__start

    @property
    def end(self) -> float:
        return self.__end

    @property
    def width(self) -> float:
        return self.end - self.start

    def contains(self, x: float) -> bool:
        return (self.start <= x) & (x < self.end)

    def discretize(self, n: Optional[int] = None, size: Optional[float] = None) -> NDArray[np.float64]:
        if n is None and size is None:
            raise ValueError("Either `n` or `size` must be specified")
        elif n is not None and size is not None:
            raise ValueError("Only one of `n` or `size` can be specified")
        elif n is not None and size is None:
            points = np.linspace(self.start, self.end, n, dtype=np.float64)
        elif n is None and size is not None:
            points = np.arange(self.start, self.end, size, dtype=np.float64)
        else:
            raise ValueError("Invalid parameters for discretization")
        return points


class Bin(Interval):
    def __init__(self, start: float, end: float, value: float):
        super().__init__(start, end)
        self.__value = value

    @property
    def value(self) -> float:
        return self.__value

