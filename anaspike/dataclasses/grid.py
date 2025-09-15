import numpy as np
from numpy.typing import NDArray

from .interval import Interval



class Grid1D:
    def __init__(self, points: NDArray[np.float64]):
        if points.ndim != 1:
            raise ValueError("points must be a one-dimensional array.")
        if len(points) < 2:
            raise ValueError("points must contain at least two points.")
        if not np.all(np.diff(points) > 0):
            raise ValueError("points must be strictly increasing.")
        
        self.__points = points

    def __array__(self) -> NDArray[np.float64]:
        return self.points

    @property
    def points(self) -> NDArray[np.float64]:
        return self.__points

    @property
    def n(self) -> int:
        return len(self.points)


class RegularGrid1D(Grid1D):
    def __init__(self, points: NDArray[np.float64], rtol: float = 1.e-5, atol: float = 1.e-8):
        if not np.all(np.isclose(np.diff(points), points[1] - points[0], rtol=rtol, atol=atol)):
            raise ValueError("`points` must be equidistant within given tolerances.")

        self._rtol = rtol
        self._atol = atol

        super().__init__(points)

    @classmethod
    def given_n(cls, interval: Interval, n: int) -> "RegularGrid1D":
        if n < 2:
            raise ValueError("n must be at least 2.")
        points = np.linspace(interval.start, interval.end, n, endpoint=False, dtype=np.float64)
        return cls(points)

    @classmethod
    def given_delta(cls, interval: Interval, delta: float) -> "RegularGrid1D":
        if delta <= 0:
            raise ValueError("`delta` must be positive.")
        points = np.arange(interval.start, interval.end, delta, dtype=np.float64)
        return cls(points)

    @property
    def delta(self) -> float:
        return self.points[1] - self.points[0]


class RectilinearGrid2D:
    def __init__(self, x: Grid1D, y: Grid1D):
        self.__x = x
        self.__y = y

    @property
    def x(self) -> Grid1D:
        return self.__x

    @property
    def y(self) -> Grid1D:
        return self.__y

    @property
    def mesh(self):
        return np.meshgrid(self.x.points, self.y.points, indexing='ij')

    @property
    def xx(self) -> NDArray[np.float64]:
        return self.mesh[0]

    @property
    def yy(self) -> NDArray[np.float64]:
        return self.mesh[1]

    @property
    def nx(self) -> int:
        return self.x.n

    @property
    def ny(self) -> int:
        return self.y.n

    @property
    def N(self) -> int:
        return self.nx * self.ny


Grid2D = RectilinearGrid2D


class RegularGrid2D(RectilinearGrid2D):
    def __init__(self, x: RegularGrid1D, y: RegularGrid1D):
        self.__x = x
        self.__y = y

    @property
    def x(self) -> RegularGrid1D:
        return self.__x

    @property
    def y(self) -> RegularGrid1D:
        return self.__y

    @property
    def delta_x(self) -> float:
        return self.x.delta

    @property
    def delta_y(self) -> float:
        return self.y.delta


class CartesianGrid2D(RectilinearGrid2D):
    def __init__(self, x: RegularGrid1D, y: RegularGrid1D, rtol: float = 1.e-5, atol: float = 1.e-8):
        if not np.isclose(x.delta, y.delta, rtol=rtol, atol=atol):
            raise ValueError("`x` and `y` must have the same delta within the given tolerances.")
        super().__init__(x, y)

