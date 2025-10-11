from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from ..dataclasses.scalar_spatio_temporal_map import ScalarSpatioTemporalMap
from ..dataclasses.coords2d import Coords2D
from ..hdf5_mixin import HDF5Mixin
from ..dataclasses.nest_devices import PopulationData, SpikeRecorderData
from ..dataclasses.histogram import Histogram
from ..dataclasses.grid import RegularGrid1D
from ..dataclasses.bins import ContigBins1D
from ..spike_trains import SpikeTrains
from ..dataclasses.scalar_temporal_map import ScalarTemporalMap



FiringRateEvolution = ScalarTemporalMap


class FiringRatesEvolution(ScalarSpatioTemporalMap[Coords2D], HDF5Mixin):
    def __init__(self, coords: Coords2D, times: NDArray[np.float64], values: NDArray[np.float64]):
        super().__init__(coords=coords, times=times, values=values)

    @classmethod
    def from_spike_trains(cls, spike_trains: SpikeTrains,
                          time_bins: ContigBins1D[RegularGrid1D]) -> "FiringRatesEvolution":
        spike_counts = np.array([Histogram.construct_by_counting(bins=time_bins, data=st).counts for st in spike_trains])
        return cls(
            coords=spike_trains.coords,
            times=time_bins.labels,
            values=np.array(spike_counts) / time_bins.edges.delta * 1.e3
        )

    @classmethod
    def from_nest(cls, pop: PopulationData, sr: SpikeRecorderData,
                  time_bins: ContigBins1D[RegularGrid1D]) -> "FiringRatesEvolution":
        spike_trains = SpikeTrains.from_nest(pop=pop, sr=sr)
        return cls.from_spike_trains(spike_trains=spike_trains, time_bins=time_bins)

    @property
    def n_neurons(self) -> int:
        return self.n_values

    def iter_neuron_dim(self) -> Iterator[ScalarTemporalMap]:
        yield from self.iter_coords_dim()

    @property
    def values_neurons_major(self) -> NDArray[np.float64]:
        return self.values_coords_major

    @property
    def _values(self):
        return self.values_coords_major

