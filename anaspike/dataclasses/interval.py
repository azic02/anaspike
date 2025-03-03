from typing import Optional, Tuple

import numpy as np



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

    def contains(self, x) -> bool:
        return (self.start <= x) & (x < self.end)

    def bin(self, n: Optional[int] = None, size: Optional[float] = None) -> Tuple['Bin', ...]:
        if n is None and size is None:
            raise ValueError("Either n or size must be specified")
        elif n is not None and size is not None:
            raise ValueError("Only one of n or size can be specified")
        elif n is not None:
            edges = np.linspace(self.start, self.end, n + 1)
        else:
            edges = np.arange(self.start, self.end, size)
        values = edges[:-1] + np.diff(edges) / 2
        return tuple(Bin(s, e, v) for s, e, v in zip(edges[:-1], edges[1:], values))


class Bin(Interval):
    def __init__(self, start: float, end: float, value: float):
        super().__init__(start, end)
        self.__value = value

    @property
    def value(self) -> float:
        return self.__value

