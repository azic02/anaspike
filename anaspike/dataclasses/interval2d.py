import numpy as np
from numpy.typing import NDArray

from .interval import Interval
from .coords2d import Coords2D



class Interval2D:
    def __init__(self, x_interval: Interval, y_interval: Interval):
        self.__x_interval = x_interval
        self.__y_interval = y_interval

    @property
    def x_interval(self) -> Interval:
        return self.__x_interval

    @property
    def y_interval(self) -> Interval:
        return self.__y_interval

    def contains(self, c: Coords2D) -> NDArray[np.bool_]:
        return self.x_interval.contains(c.x) & self.y_interval.contains(c.y)

