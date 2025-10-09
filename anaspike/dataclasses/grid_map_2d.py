from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from .grid import RectilinearGrid2D, RegularGrid2D, RegularGrid1D



ElmnT = TypeVar("ElmnT", bound=np.generic)
GridT = TypeVar("GridT", bound=RectilinearGrid2D)
class GridMap2D(Generic[GridT, ElmnT]):
    def __init__(self, grid: GridT, elements: NDArray[ElmnT]):
        if elements.ndim < 2:
            raise ValueError("elements must be at least a two-dimensional array.")
        if grid.shape != elements.shape[:2]:
            raise ValueError(f"shape of grid {grid.shape} must match first two dimensions of elements {elements.shape[:2]}.")
        
        self.__grid = grid
        self.__elements = elements

    @property
    def grid(self) -> GridT:
        return self.__grid

    @property
    def elements(self) -> NDArray[ElmnT]:
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

    def ravel_index(self, ix: np.intp, iy: np.intp):
        return np.ravel_multi_index((ix, iy), self.grid.shape)

    def unravel_index(self, i: np.intp):
        return np.unravel_index(i, self.grid.shape)


def calculate_fft_2d(gf: GridMap2D[RegularGrid2D,ElmnT]):
    freq_x = np.fft.fftshift(np.fft.fftfreq(gf.nx, d=gf.grid.delta_x)).astype(np.float64)
    freq_y = np.fft.fftshift(np.fft.fftfreq(gf.ny, d=gf.grid.delta_y)).astype(np.float64)
    grid = RegularGrid2D(x=RegularGrid1D(freq_x), y=RegularGrid1D(freq_y))
    fft_elements = np.fft.fftshift(np.fft.fft2(gf.elements))
    return GridMap2D(grid, fft_elements)


def calculate_ifft_2d(gf: GridMap2D[RegularGrid2D,ElmnT]):
    spatial_x = np.fft.fftshift(np.fft.fftfreq(gf.nx, d=gf.grid.delta_x))
    spatial_y = np.fft.fftshift(np.fft.fftfreq(gf.ny, d=gf.grid.delta_y))
    grid = RegularGrid2D(
        x=RegularGrid1D(spatial_x.astype(np.float64)),
        y=RegularGrid1D(spatial_y.astype(np.float64))
    )
    
    ifft_data = np.fft.ifft2(np.fft.ifftshift(gf.elements))
    return GridMap2D(grid, ifft_data)


def calculate_psd_2d(gf: GridMap2D[RegularGrid2D,ElmnT]) -> GridMap2D[RegularGrid2D,np.float64]:
    fft = calculate_fft_2d(gf)
    psd_elements = np.abs(fft.elements) ** 2
    return GridMap2D(fft.grid, psd_elements)


def calculate_autocorrelation_2d_wiener_khinchin(gf: GridMap2D[RegularGrid2D,ElmnT]) -> GridMap2D[RegularGrid2D,np.float64]:
    psd = calculate_psd_2d(gf)
    unshifted_ac = calculate_ifft_2d(psd)
    ac_at_zero_lag = unshifted_ac.elements[0,0].real
    normalised_ac = GridMap2D(unshifted_ac.grid, np.fft.fftshift(unshifted_ac.elements.real) / ac_at_zero_lag)
    return normalised_ac

