import numpy as np
from numpy.typing import NDArray, DTypeLike
from typing import Union, Collection

from anaspike.hdf5_mixin import HDF5Mixin
from anaspike.dataclasses.nest_devices import SpikeRecorderData, PopulationData



SpikeTrain = NDArray[np.float64]

class SpikeTrainArray(HDF5Mixin):
    def __init__(self, spike_trains: Collection[SpikeTrain]):
        self._spike_trains = np.asarray(spike_trains, dtype=object)

    @classmethod
    def from_nest(cls, pop: PopulationData, sr: SpikeRecorderData) -> 'SpikeTrainArray':
        return cls([np.sort(sr.times[sr.senders == i]) for i in pop.ids])

    @property
    def n_neurons(self) -> int:
        return self._spike_trains.shape[0]

    def __array__(self, dtype: Union[DTypeLike, None] = None):
        return np.asarray(self._spike_trains, dtype=dtype)

    def __getitem__(self, index: Union[int, slice, NDArray[np.int64]]) -> Union['SpikeTrainArray', SpikeTrain]:
        if isinstance(index, int):
            return np.asarray(self._spike_trains[index], dtype=np.float64)
        return SpikeTrainArray(self._spike_trains[index])

    def __len__(self):
        return self.n_neurons

    def __iter__(self):
        for train in self._spike_trains:
            yield np.asarray(train, dtype=np.float64)

    @property
    def dtype(self):
        return self._spike_trains.dtype

