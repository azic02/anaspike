from typing import TypeVar, Generic, Tuple

import numpy as np
from numpy.typing import NDArray

from ..hdf5_mixin import HDF5Mixin
from .grid import Grid1D, RegularGrid1D, RectilinearGrid2D
from .interval import Interval
from .coords2d import Coords2D
from .cartesian_map_2d import CartesianMap2D
from .grid_map_2d import  GridMap2D
from .interval import Bin



Grid1dT = TypeVar("Grid1dT", bound=Grid1D)
class ContigBins1D(Generic[Grid1dT], HDF5Mixin):
    def __init__(self, grid: Grid1dT, labels: NDArray[np.float64]):
        if labels.ndim != 1:
            raise ValueError("'labels' must be a 1D array.")
        if grid.n != labels.shape[0] + 1:
            raise ValueError("Number of 'grid' points must be one more than number of labels.")
        self.__grid = grid
        self.__labels = labels

    @classmethod
    def with_median_labels(cls, grid: Grid1dT):
        labels = grid[:-1] + np.diff(grid) / 2
        return cls(grid, labels)

    @classmethod
    def from_str(cls, spec: str) -> "ContigBins1D[RegularGrid1D]":
        try:
            parts = spec.split(',')
            if len(parts) != 3:
                raise ValueError("Specification string must have three parts separated by commas.")
            start = float(parts[0])
            end = float(parts[1])
            if parts[2].startswith('n'):
                n = int(parts[2][1:])
                return ContigBins1D[RegularGrid1D].with_median_labels(
                        RegularGrid1D.from_interval_given_n(Interval(start, end),
                                                            n + 1,
                                                            endpoint=True))
            elif parts[2].startswith('d'):
                delta = float(parts[2][1:])
                return ContigBins1D[RegularGrid1D].with_median_labels(
                        RegularGrid1D.from_interval_given_delta(Interval(start, end), delta))
            else:
                raise ValueError("Third part of specification must start with 'n' or 'd'.")
        except Exception as e:
            raise ValueError(f"Invalid specification string: {spec}. Interval string must be in the format '<start>,<end>,n<n_bins>' or '<start>,<end>,d<delta>'") from e

    @property
    def labels(self) -> NDArray[np.float64]:
        return self.__labels

    @property
    def grid(self) -> Grid1dT:
        return self.__grid

    @property
    def edges(self) -> Grid1dT:
        return self.grid

    @property
    def n(self) -> int:
        return len(self.labels)

    def __len__(self) -> int:
        return len(self.__labels)

    def __getitem__(self, idx: int) -> Bin:
        if idx >= 0 and idx < len(self):
            edge_idx = idx
        elif idx < 0 and idx >= -len(self):
            edge_idx = idx - 1
        else:
            raise IndexError("Index out of bounds")
        return Bin(start=self.grid[edge_idx],
                   end=self.grid[edge_idx + 1],
                   value=self.labels[idx])

    def __iter__(self):
        return (self[i] for i in range(len(self.labels)))


Grid2dT = TypeVar("Grid2dT", bound=RectilinearGrid2D)
class ContigBins2D(Generic[Grid2dT]):
    def __init__(self, grid: Grid2dT, labels: NDArray[np.float64]):
        wanted_labels_shape = (grid.nx - 1, grid.ny - 1, 2)
        if labels.shape != wanted_labels_shape:
            raise ValueError(f"`labels` must have shape `(grid.nx - 1, grid.ny - 1, 2) = {wanted_labels_shape}`. Got `{labels.shape} instead.`")

        self.__grid = grid
        self.__labels = labels

    @classmethod
    def given_x_y_labels(cls, grid: Grid2dT, x_labels: NDArray[np.float64], y_labels: NDArray[np.float64]):
        if x_labels.ndim != 1 or y_labels.ndim != 1:
            raise ValueError("x_labels and y_labels must be 1D arrays.")
        if grid.nx != x_labels.shape[0] + 1:
            raise ValueError("Number of `grid.x` points must be one more than number of `x_labels`.")
        if grid.ny != y_labels.shape[0] + 1:
            raise ValueError("Number of `grid.y` points must be one more than number of `y_labels`.")
        labels = np.array([[(x, y) for y in y_labels] for x in x_labels])
        return cls(grid, labels)

    @classmethod
    def with_median_labels(cls, grid: Grid2dT):
        x_labels = grid.x[:-1] + np.diff(grid.x) / 2
        y_labels = grid.y[:-1] + np.diff(grid.y) / 2
        return cls.given_x_y_labels(grid, x_labels, y_labels)

    @property
    def grid(self) -> Grid2dT:
        return self.__grid

    @property
    def labels(self) -> NDArray[np.float64]:
        return self.__labels

    @property
    def x_edges(self) -> Grid1D:
        return self.grid.x

    @property
    def y_edges(self) -> Grid1D:
        return self.grid.y

    @property
    def x_labels(self) -> NDArray[np.float64]:
        return self.labels[:, 0, 0]

    @property
    def y_labels(self) -> NDArray[np.float64]:
        return self.labels[0, :, 1]

    @property
    def x(self) -> ContigBins1D[Grid1D]:
        return ContigBins1D(self.x_edges, self.x_labels)

    @property
    def y(self) -> ContigBins1D[Grid1D]:
        return ContigBins1D(self.y_edges, self.y_labels)

    @property
    def nx(self) -> int:
        return self.x.n

    @property
    def ny(self) -> int:
        return self.y.n

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.nx, self.ny)


def assign_bins_2d(bins: ContigBins2D[Grid2dT], coords: Coords2D) -> Tuple[NDArray[np.int64], NDArray[np.int64]]: #TODO: returns Coords2D[np.int64]
    x_bin_idxs = np.digitize(coords.x, bins.x_edges) - 1
    y_bin_idxs = np.digitize(coords.y, bins.y_edges) - 1
    if np.any(x_bin_idxs < 0) or np.any(x_bin_idxs >= bins.nx) or np.any(y_bin_idxs < 0) or np.any(y_bin_idxs >= bins.ny):
        raise ValueError("Some labels are outside the bin edges.")
    return x_bin_idxs, y_bin_idxs

def calculate_bin_counts_2d(bins: ContigBins2D[Grid2dT], coords: Coords2D) -> GridMap2D[Grid2dT,np.int64]:
    #bin_counts, _, _ = np.histogram2d(coords.x, coords.y, bins=[bins.x.edges, bins.y.edges])
    x_bin_idxs, y_bin_idxs = assign_bins_2d(bins, coords)
    bin_counts = np.zeros(shape=bins.shape, dtype=np.int64)
    np.add.at(bin_counts, (x_bin_idxs, y_bin_idxs), 1)
    return GridMap2D(bins.grid.__class__(bins.grid.x.__class__(bins.x_labels), bins.grid.y.__class__(bins.y_labels)), bin_counts)

def calculate_bin_sums_2d(bins: ContigBins2D[Grid2dT], field: CartesianMap2D[np.float64]) -> GridMap2D[Grid2dT,np.float64]:
    x_bin_idxs, y_bin_idxs = assign_bins_2d(bins, field.coords)
    bin_sums = np.zeros(shape=(*bins.shape, *field.element_shape), dtype=np.float64)
    np.add.at(bin_sums, (x_bin_idxs, y_bin_idxs), field.elements)
    return GridMap2D(bins.grid.__class__(bins.grid.x.__class__(bins.x_labels), bins.grid.y.__class__(bins.y_labels)), bin_sums)

def calculate_bin_means_2d(bins: ContigBins2D[Grid2dT], field:
                        CartesianMap2D[np.float64]) -> GridMap2D[Grid2dT,np.float64]:
    bin_sums = calculate_bin_sums_2d(bins, field).elements
    bin_counts = calculate_bin_counts_2d(bins, field.coords).elements
    broadcasted_bin_counts = bin_counts[..., np.newaxis] if field.element_ndim > 0 else bin_counts
    bin_means = np.divide(bin_sums, broadcasted_bin_counts, out=np.full_like(bin_sums, np.nan), where=broadcasted_bin_counts != 0)
    return GridMap2D(bins.grid.__class__(bins.grid.x.__class__(bins.x_labels), bins.grid.y.__class__(bins.y_labels)), bin_means)


