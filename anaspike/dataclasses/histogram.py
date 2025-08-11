from typing import Optional, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray
import h5py
from matplotlib.axes import Axes

from .interval import Interval, Bin



class ContigBins:
    def __init__(self, edges: NDArray[np.float64], values: NDArray[np.float64]):
        if edges.ndim != 1 or values.ndim != 1:
            raise ValueError("edges and values must be 1D arrays")
        if len(edges) < 2:
            raise ValueError("At least two bin edges are required to create bins.")
        if len(edges) != len(values) + 1:
            raise ValueError("Number of edges must be one more than number of values.")
        if not np.all(np.diff(edges) > 0):
            raise ValueError("Bin edges must be strictly increasing.")

        self.__edges = edges
        self.__values = values
        
    @classmethod
    def from_bin_sequence(cls, bins: Sequence[Bin], max_gap: Optional[float] = 1.e-8, max_overlap: Optional[float] = 1.e-8):
        bin_starts = np.array([b.start for b in bins])
        if max_gap is not None or max_overlap is not None:
            bin_ends = np.array([b.end for b in bins])
            bin_gaps = bin_ends[:-1] - bin_starts[1:]
            if max_gap is not None:
                if max_gap < 0:
                    raise ValueError("max_gap must be non-negative")
                elif np.any(bin_gaps > max_gap):
                    raise ValueError("Bins contain gaps larger than max_gap")
            if max_overlap is not None:
                if max_overlap < 0:
                    raise ValueError("max_overlap must be non-negative")
                else:
                    min_gap = -max_overlap
                    if np.any(bin_gaps < min_gap):
                        raise ValueError("Bins contain overlaps larger than max_overlap")

        return cls(edges=np.append(bin_starts, bins[-1].end),
                   values=np.array([b.value for b in bins]))

    @classmethod
    def with_median_values(cls, edges: NDArray[np.float64]):
        values = edges[:-1] + np.diff(edges) / 2
        return cls(edges, values)

    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group):
        edges = np.array(hdf5_group["edges"][:])
        values = np.array(hdf5_group["values"][:])
        return cls(edges, values)

    def to_hdf5(self, hdf5_group: h5py.Group):
        hdf5_group.create_dataset("edges", data=self.__edges)
        hdf5_group.create_dataset("values", data=self.__values)

    @property
    def bin_edges(self) -> NDArray[np.float64]:
        return self.__edges

    @property
    def edges(self) -> NDArray[np.float64]:
        return self.bin_edges

    @property
    def bin_values(self) -> NDArray[np.float64]:
        return self.__values

    @property
    def values(self) -> NDArray[np.float64]:
        return self.values

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

    def __iter__(self) -> Iterator[Bin]:
        return (self[i] for i in range(len(self.__values)))


class EquiBins(ContigBins):
    def __init__(self, edges: NDArray[np.float64], values: NDArray[np.float64], rtol: float = 1.e-5, atol: float = 1.e-8):
        if not np.all(np.isclose(np.diff(edges), edges[1] - edges[0], rtol=rtol, atol=atol)):
            raise ValueError("All bins must have the same width.")

        super().__init__(edges, values)

    @classmethod
    def from_interval_with_median_values(cls, interval: Interval, n: Optional[int] = None, size: Optional[float] = None):
        n_edges = None if n is None else n + 1
        edges = interval.discretize(n_edges, size)
        return cls.with_median_values(edges)

    @classmethod
    def from_interval_str_with_median_values(cls, equi_bins_str: str):
        parts = equi_bins_str.split(',')
        if len(parts) != 3:
            raise ValueError("Interval string must be in the format '<start>,<end>,n<n_bins>' or '<start>,<end>,s<bin_size>'")
        interval_str = ','.join(parts[:2])
        interval = Interval.from_str(interval_str)

        bin_spec_str = parts[2]
        if len(bin_spec_str) <= 1:
            raise ValueError("Invalid bin specification")
        if bin_spec_str[0] == 'n':
            n = int(bin_spec_str[1:])
            size = None
        elif bin_spec_str[0] == 's':
            n = None
            size = float(bin_spec_str[1:])
        else:
            raise ValueError("Bin specification must start with 'n' or 's'")
        return cls.from_interval_with_median_values(interval, n, size)

    @property
    def bin_width(self) -> float:
        return self[0].width


class Histogram:
    def __init__(self, bins: ContigBins, counts: NDArray[np.int64]):
        if len(bins) != len(counts):
            raise ValueError("Length of bins must match length of counts.")

        self.__bins = bins
        self.__counts = counts

    @classmethod
    def construct_by_counting(cls, bins: ContigBins, data: NDArray[np.float64]):
        counts, _ = np.histogram(data, bins=bins.bin_edges)
        return cls(bins, counts)

    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group):
        bins = ContigBins.from_hdf5(hdf5_group["bins"])
        counts = np.array(hdf5_group["counts"][:])
        return cls(bins, counts)

    def to_hdf5(self, hdf5_group: h5py.Group):
        bins_group = hdf5_group.create_group("bins")
        self.__bins.to_hdf5(bins_group)
        hdf5_group.create_dataset("counts", data=self.__counts)

    @property
    def bins(self) -> ContigBins:
        return self.__bins

    @property
    def bin_edges(self) -> NDArray[np.float64]:
        return self.__bins.bin_edges

    @property
    def bin_values(self) -> NDArray[np.float64]:
        return self.__bins.bin_values

    @property
    def counts(self) -> NDArray[np.int64]:
        return self.__counts

    def plot(self, ax: Axes, **kwargs):
        ax.bar(self.bin_values, self.counts, width=np.diff(self.bin_edges), align='center', **kwargs)
        ax.set_ylabel('Counts')
        return ax

