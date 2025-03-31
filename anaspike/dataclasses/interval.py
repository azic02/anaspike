from typing import Iterator, Optional, Sequence, Tuple

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

    def discretize(self, n: Optional[int] = None, size: Optional[float] = None) -> np.ndarray:
        if n is None and size is None:
            raise ValueError("Either n or size must be specified")
        elif n is not None and size is not None:
            raise ValueError("Only one of n or size can be specified")
        elif n is not None:
            points = np.linspace(self.start, self.end, n)
        else:
            points = np.arange(self.start, self.end, size)
        return points

    def bin(self, n: Optional[int] = None, size: Optional[float] = None) -> 'EquidistantBins':
        return EquidistantBins(self, n, size)


class Bin(Interval):
    def __init__(self, start: float, end: float, value: float):
        super().__init__(start, end)
        self.__value = value

    @property
    def value(self) -> float:
        return self.__value


class Bins:
    def __init__(self, bins: Sequence[Bin]):
        self.__bins = bins

    def __getitem__(self, key: int) -> Bin:
        return self.__bins[key]

    def __len__(self) -> int:
        return len(self.__bins)

    def __iter__(self) -> Iterator[Bin]:
        return iter(self.__bins)


class EquidistantBins(Bins):
    def __init__(self, interval: Interval, n: Optional[int] = None, size: Optional[float] = None):
        if n is not None:
            n += 1
        edges = interval.discretize(n, size)
        values = edges[:-1] + np.diff(edges) / 2
        super().__init__(tuple(Bin(s, e, v) for s, e, v in zip(edges[:-1], edges[1:], values)))

    @property
    def bin_width(self) -> float:
        return self[0].width

