from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from matplotlib.axes import Axes

from ..hdf5_mixin import HDF5Mixin
from .bins import ContigBins1D
from .grid import Grid1D



Grid1dT = TypeVar("Grid1dT", bound=Grid1D)
class Histogram(HDF5Mixin):
    def __init__(self, bins: ContigBins1D[Grid1dT], counts: NDArray[np.int64]):
        if bins.n != len(counts):
            raise ValueError("Length of bins must match length of counts.")

        self.__bins = bins
        self.__counts = counts

    @classmethod
    def construct_by_counting(cls, bins: ContigBins1D[Grid1dT], data: NDArray[np.float64]):
        counts, _ = np.histogram(data, bins=bins.edges)
        return cls(bins, counts)

    @property
    def bins(self):
        return self.__bins

    @property
    def edges(self):
        return self.__bins.edges

    @property
    def labels(self) -> NDArray[np.float64]:
        return self.__bins.labels

    @property
    def counts(self) -> NDArray[np.int64]:
        return self.__counts

    def plot(self, ax: Axes, **kwargs):
        ax.bar(self.labels, self.counts, width=np.diff(self.edges), align='center', **kwargs)
        ax.set_ylabel('Counts')
        return ax

