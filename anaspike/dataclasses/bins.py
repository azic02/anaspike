from typing import TypeVar, Generic, Tuple

import numpy as np
from numpy.typing import NDArray

from ..hdf5_mixin import HDF5Mixin
from .grid import Grid1D, RectilinearGrid2D
from .coords2d import Coords2D
from .field import Field2D, GridField2D
from .interval import Bin



Grid1dT = TypeVar("Grid1dT", bound=Grid1D)
class ContigBins1D(Generic[Grid1dT], HDF5Mixin):
    def __init__(self, edges: Grid1dT, values: NDArray[np.float64]):
        if values.ndim != 1:
            raise ValueError("values must be a 1D array.")
        if edges.n != values.shape[0] + 1:
            raise ValueError("Number of edges must be one more than number of values.")
        self.__edges = edges
        self.__values = values


    @classmethod
    def with_median_values(cls, edges: Grid1dT):
        values = edges[:-1] + np.diff(edges) / 2
        return cls(edges, values)

    @property
    def edges(self) -> Grid1dT:
        return self.__edges

    @property
    def values(self) -> NDArray[np.float64]:
        return self.__values

    @property
    def n(self) -> int:
        return len(self.values)

    def __len__(self) -> int:
        return len(self.__values)

    def __getitem__(self, idx: int) -> Bin:
        if idx >= 0 and idx < len(self):
            edge_idx = idx
        elif idx < 0 and idx >= -len(self):
            edge_idx = idx - 1
        else:
            raise IndexError("Index out of bounds")
        return Bin(start=self.__edges[edge_idx],
                   end=self.__edges[edge_idx + 1],
                   value=self.__values[idx])

    def __iter__(self):
        return (self[i] for i in range(len(self.__values)))


Grid2dT = TypeVar("Grid2dT", bound=RectilinearGrid2D)
class ContigBins2D(Generic[Grid2dT]):
    def __init__(self, edges: Grid2dT, values: NDArray[np.float64]):
        wanted_values_shape = (edges.nx - 1, edges.ny - 1, 2)
        if values.shape != wanted_values_shape:
            raise ValueError(f"`values` must have shape `(edges.nx - 1, edges.ny - 1, 2) = {wanted_values_shape}`. Got `{values.shape} instead.`")

        self.__edges = edges
        self.__values = values

    @classmethod
    def given_x_y_values(cls, edges: Grid2dT, x_values: NDArray[np.float64], y_values: NDArray[np.float64]):
        if x_values.ndim != 1 or y_values.ndim != 1:
            raise ValueError("x_values and y_values must be 1D arrays.")
        if edges.nx != x_values.shape[0] + 1:
            raise ValueError("Number of x edges must be one more than number of x values.")
        if edges.ny != y_values.shape[0] + 1:
            raise ValueError("Number of y edges must be one more than number of y values.")
        values = np.array([[(x, y) for y in y_values] for x in x_values])
        return cls(edges, values)

    @classmethod
    def with_median_values(cls, edges: Grid2dT):
        x_values = edges.x[:-1] + np.diff(edges.x) / 2
        y_values = edges.y[:-1] + np.diff(edges.y) / 2
        return cls.given_x_y_values(edges, x_values, y_values)


    @property
    def edges(self) -> Grid2dT:
        return self.__edges

    @property
    def values(self) -> NDArray[np.float64]:
        return self.__values

    @property
    def x(self) -> ContigBins1D[Grid1D]:
        return ContigBins1D(self.edges.x, self.values[:,0,0])

    @property
    def y(self) -> ContigBins1D[Grid1D]:
        return ContigBins1D(self.edges.y, self.values[0,:,1])

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
    x_bin_idxs = np.digitize(coords.x, bins.edges.x) - 1
    y_bin_idxs = np.digitize(coords.y, bins.y.edges) - 1
    if np.any(x_bin_idxs < 0) or np.any(x_bin_idxs >= bins.nx) or np.any(y_bin_idxs < 0) or np.any(y_bin_idxs >= bins.ny):
        raise ValueError("Some values are outside the bin edges.")
    return x_bin_idxs, y_bin_idxs

def calculate_bin_counts_2d(bins: ContigBins2D[Grid2dT], coords: Coords2D) -> GridField2D[Grid2dT,np.int64]:
    #bin_counts, _, _ = np.histogram2d(coords.x, coords.y, bins=[bins.x.edges, bins.y.edges])
    x_bin_idxs, y_bin_idxs = assign_bins_2d(bins, coords)
    bin_counts = np.zeros(shape=bins.shape, dtype=np.int64)
    np.add.at(bin_counts, (x_bin_idxs, y_bin_idxs), 1)
    return GridField2D(bins.edges.__class__(Grid1D(bins.x.values), Grid1D(bins.y.values)), bin_counts)

def calculate_bin_sums_2d(bins: ContigBins2D[Grid2dT], field: Field2D[np.float64]) -> GridField2D[Grid2dT,np.float64]:
    x_bin_idxs, y_bin_idxs = assign_bins_2d(bins, field.coords)
    bin_sums = np.zeros(shape=(*bins.shape, *field.element_shape), dtype=np.float64)
    np.add.at(bin_sums, (x_bin_idxs, y_bin_idxs), field.elements)
    return GridField2D(bins.edges.__class__(Grid1D(bins.x.values), Grid1D(bins.y.values)), bin_sums)

def calculate_bin_means_2d(bins: ContigBins2D[Grid2dT], field:
                        Field2D[np.float64]) -> GridField2D[Grid2dT,np.float64]:
    bin_sums = calculate_bin_sums_2d(bins, field).elements
    bin_counts = calculate_bin_counts_2d(bins, field.coords).elements
    broadcasted_bin_counts = bin_counts[..., np.newaxis] if field.element_ndim > 0 else bin_counts
    bin_means = np.divide(bin_sums, broadcasted_bin_counts, out=np.full_like(bin_sums, np.nan), where=broadcasted_bin_counts != 0)
    return GridField2D(bins.edges.__class__(Grid1D(bins.x.values), Grid1D(bins.y.values)), bin_means)


