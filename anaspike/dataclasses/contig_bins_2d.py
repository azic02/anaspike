from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from .histogram import ContigBins
from .coords2d import Coords2D
from .grid import RectilinearGrid2D, Grid1D
from .field import Field2D, GridField2D



class ContigBins2D:
    def __init__(self, x: ContigBins, y: ContigBins):
        self.__x = x
        self.__y = y

    def __len__(self) -> int:
        return len(self.__x) * len(self.__y)

    @property
    def x(self) -> ContigBins:
        return self.__x

    @property
    def y(self) -> ContigBins:
        return self.__y


def assign_bins(bins: ContigBins2D, coords: Coords2D) -> Tuple[NDArray[np.int64], NDArray[np.int64]]: #TODO: returns Coords2D[np.int64]
    x_bin_idxs = np.digitize(coords.x, bins.x.edges) - 1
    y_bin_idxs = np.digitize(coords.y, bins.y.edges) - 1
    if np.any(x_bin_idxs < 0) or np.any(x_bin_idxs >= len(bins.x)) or np.any(y_bin_idxs < 0) or np.any(y_bin_idxs >= len(bins.y)):
        raise ValueError("Some values are outside the bin edges.")
    return x_bin_idxs, y_bin_idxs

def calculate_bin_counts(bins: ContigBins2D, coords: Coords2D) -> GridField2D[RectilinearGrid2D,np.int64]:
    #bin_counts, _, _ = np.histogram2d(coords.x, coords.y, bins=[bins.x.edges, bins.y.edges])
    x_bin_idxs, y_bin_idxs = assign_bins(bins, coords)
    bin_counts = np.zeros(shape=(len(bins.x), len(bins.y)), dtype=np.int64)
    np.add.at(bin_counts, (x_bin_idxs, y_bin_idxs), 1)
    return GridField2D(RectilinearGrid2D(Grid1D(bins.x.values), Grid1D(bins.y.values)), bin_counts)

def calculate_bin_sums(bins: ContigBins2D, field: Field2D[np.float64]) -> GridField2D[RectilinearGrid2D,np.float64]:
    x_bin_idxs, y_bin_idxs = assign_bins(bins, field.coords)
    bin_sums = np.zeros(shape=(len(bins.x), len(bins.y), *field.element_shape), dtype=np.float64)
    np.add.at(bin_sums, (x_bin_idxs, y_bin_idxs), field.elements)
    return GridField2D(RectilinearGrid2D(Grid1D(bins.x.values), Grid1D(bins.y.values)), bin_sums)

def calculate_bin_means(bins: ContigBins2D, field: Field2D[np.float64]) -> GridField2D[RectilinearGrid2D,np.float64]:
    bin_sums = calculate_bin_sums(bins, field).elements
    bin_counts = calculate_bin_counts(bins, field.coords).elements
    broadcasted_bin_counts = bin_counts[..., np.newaxis] if field.element_ndim > 0 else bin_counts
    bin_means = np.divide(bin_sums, broadcasted_bin_counts, out=np.full_like(bin_sums, np.nan), where=broadcasted_bin_counts != 0)
    return GridField2D(RectilinearGrid2D(Grid1D(bins.x.values), Grid1D(bins.y.values)), bin_means)
