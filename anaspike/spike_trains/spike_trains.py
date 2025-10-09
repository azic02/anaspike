from typing import overload, Union, Collection

import numpy as np
from numpy.typing import NDArray, DTypeLike

from anaspike.dataclasses.cartesian_map_2d import CartesianMap2D
from anaspike.hdf5_mixin import HDF5Mixin
from anaspike.dataclasses.coords2d import Coords2D
from anaspike.dataclasses.nest_devices import SpikeRecorderData, PopulationData



SpikeTrain = NDArray[np.float64]

class SpikeTrains(CartesianMap2D[np.object_], HDF5Mixin):
    def __init__(self, coords: Coords2D, spike_trains: Collection[SpikeTrain]):
        spike_trains = np.asarray(spike_trains, dtype=object)
        super().__init__(coords, spike_trains)

    @classmethod
    def from_nest(cls, pop: PopulationData, sr: SpikeRecorderData) -> 'SpikeTrains':
        spike_trains = [np.sort(sr.times[sr.senders == i]) for i in pop.ids]
        return cls(pop.coords, spike_trains)

    @property
    def spike_trains(self):
        return self.elements

    @property
    def n_neurons(self) -> int:
        return self.spike_trains.shape[0]

    def __array__(self, dtype: Union[DTypeLike, None] = None):
        return np.asarray(self.spike_trains, dtype=dtype)

    @overload
    def __getitem__(self, s: int) -> SpikeTrain: ...
    @overload
    def __getitem__(self, s: slice) -> 'SpikeTrains': ...
    @overload
    def __getitem__(self, s: NDArray[np.int64]) -> 'SpikeTrains': ...
    def __getitem__(self, s: Union[int, slice, NDArray[np.int64]]) -> Union['SpikeTrains', SpikeTrain]:
        if isinstance(s, int):
            return np.asarray(self.spike_trains[s], dtype=np.float64)
        return SpikeTrains(self.coords[s], self.spike_trains[s])

    def __len__(self):
        return self.n_neurons

    def __iter__(self):
        for train in self.spike_trains:
            yield np.asarray(train, dtype=np.float64)

    @property
    def dtype(self):
        return self.spike_trains.dtype

