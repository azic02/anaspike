from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from .coords2d import Coords2D
from .grid import RectilinearGrid2D, RegularGrid2D, RegularGrid1D



SigT = TypeVar("SigT", bound=np.generic)
class Field2D(Generic[SigT]):
    def __init__(self, coords: Coords2D, elements: NDArray[SigT]):
        if elements.ndim < 1:
            raise ValueError("elements must at least be a one-dimensional array.")
        if len(coords) != elements.shape[0]:
            raise ValueError("length of coords must match length of elements.")
       
        self.__coords = coords
        self.__elements = elements

    @property
    def coords(self) -> Coords2D:
        return self.__coords

    @property
    def elements(self) -> NDArray[SigT]:
        return self.__elements

    @property
    def n(self) -> int:
        return self.elements.shape[0]

    @property
    def element_shape(self):
        return self.elements.shape[1:] if self.elements.ndim > 1 else ()

    @property
    def element_ndim(self) -> int:
        return self.elements.ndim - 1


GridT = TypeVar("GridT", bound=RectilinearGrid2D)
class GridField2D(Generic[GridT, SigT]):
    def __init__(self, grid: GridT, elements: NDArray[SigT]):
        if elements.ndim < 2:
            raise ValueError("elements must be at least a two-dimensional array.")
        if grid.N != elements.shape[0] * elements.shape[1]:
            raise ValueError("number of points in coordinate grid (`grid.N`) must match the number of elements in `elements`.")
        
        self.__grid = grid
        self.__elements = elements

    @property
    def grid(self) -> GridT:
        return self.__grid

    @property
    def elements(self) -> NDArray[SigT]:
        return self.__elements

    @property
    def nx(self) -> int:
        return self.elements.shape[0]

    @property
    def ny(self) -> int:
        return self.elements.shape[1]

    @property
    def mesh(self):
        return self.grid.mesh

    @property
    def xx(self) -> NDArray[np.float64]:
        return self.grid.xx

    @property
    def yy(self) -> NDArray[np.float64]:
        return self.grid.yy


def calculate_psd_2d(elements: GridField2D[RegularGrid2D,np.float64]) -> GridField2D[RegularGrid2D,np.float64]:
    freq_x = np.fft.fftshift(np.fft.fftfreq(elements.nx, d=elements.grid.delta_x)).astype(np.float64)
    freq_y = np.fft.fftshift(np.fft.fftfreq(elements.ny, d=elements.grid.delta_y)).astype(np.float64)
    grid = RegularGrid2D(x=RegularGrid1D(freq_x), y=RegularGrid1D(freq_y))

    Z_fft = np.fft.fft2(elements.elements)
    Z_shifted = np.fft.fftshift(Z_fft)
    psd = np.abs(Z_shifted)**2 / (elements.nx**2 * elements.ny**2)  # Normalized power
    return GridField2D(grid, psd)

