from typing import Union, overload

import numpy as np
from numpy.typing import NDArray

from .interval import Interval
from ..hdf5_mixin import HDF5Mixin



class Grid1D(HDF5Mixin):
    def __init__(self, points: NDArray[np.float64]):
        if points.ndim != 1:
            raise ValueError("points must be a one-dimensional array.")
        if not np.all(np.diff(points) > 0):
            raise ValueError("points must be strictly increasing.")
        
        self.__points = points

    def __array__(self) -> NDArray[np.float64]:
        return self.points

        return self.points[s]

    @property
    def points(self) -> NDArray[np.float64]:
        return self.__points

    @property
    def n(self) -> int:
        return len(self.points)

    def __len__(self) -> int:
        return len(self.points)

    @overload
    def __getitem__(self, s: int) -> float: ...
    @overload
    def __getitem__(self, s: slice) -> NDArray[np.float64]: ...
    def __getitem__(self, s: Union[int, slice]) -> Union[float, NDArray[np.float64]]:
        return self.points[s]

    def __iter__(self):
        return (self[i] for i in range(len(self.points)))


class RegularGrid1D(Grid1D):
    def __init__(self, points: NDArray[np.float64], rtol: float = 1.e-5, atol: float = 1.e-8):
        if not np.all(np.isclose(np.diff(points), points[1] - points[0], rtol=rtol, atol=atol)):
            raise ValueError("`points` must be equidistant within given tolerances.")

        self._rtol = rtol
        self._atol = atol

        super().__init__(points)

    @classmethod
    def from_interval_given_n(cls, interval: Interval, n: int, endpoint: bool) -> "RegularGrid1D":
        if n < 2:
            raise ValueError("n must be at least 2.")
        points = np.linspace(interval.start, interval.end, n, endpoint=endpoint, dtype=np.float64)
        return cls(points)

    @classmethod
    def from_interval_given_delta(cls, interval: Interval, delta: float) -> "RegularGrid1D":
        if delta <= 0:
            raise ValueError("`delta` must be positive.")
        points = np.arange(interval.start, interval.end, delta, dtype=np.float64)
        return cls(points)

    @classmethod
    def from_str(cls, spec: str) -> "RegularGrid1D":
        try:
            parts = spec.split(',')
            if len(parts) != 3:
                raise ValueError("Specification string must have three parts separated by commas.")
            start = float(parts[0])
            end = float(parts[1])
            if parts[2].startswith('n'):
                n = int(parts[2][1:])
                return cls.from_interval_given_n(Interval(start, end), n, endpoint=True)
            elif parts[2].startswith('d'):
                delta = float(parts[2][1:])
                return cls.from_interval_given_delta(Interval(start, end), delta)
            else:
                raise ValueError("Third part of specification must start with 'n' or 'd'.")
        except Exception as e:
            raise ValueError(f"Invalid specification string: {spec}. Interval string must be in the format '<start>,<end>,n<n_bins>' or '<start>,<end>,d<delta>'") from e

    @property
    def delta(self) -> float:
        return self.points[1] - self.points[0]


class RectilinearGrid2D(HDF5Mixin):
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

