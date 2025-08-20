import numpy as np
from numpy.typing import NDArray, DTypeLike
from typing import Union, Collection

from anaspike.hdf5_mixin import HDF5Mixin



SpikeTrain = NDArray[np.float64]

class SpikeTrainArray(HDF5Mixin):
    def __init__(self, spike_trains: Collection[SpikeTrain]):
        self._data = np.asarray(spike_trains, dtype=object)

    def __array__(self, dtype: Union[DTypeLike, None] = None):
        return np.asarray(self._data, dtype=dtype)

    def __getitem__(self, index: Union[int, slice, NDArray[np.int64]]) -> Union['SpikeTrainArray', SpikeTrain]:
        if isinstance(index, int):
            return np.asarray(self._data[index], dtype=np.float64)
        return SpikeTrainArray(self._data[index])

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for train in self._data:
            yield np.asarray(train, dtype=np.float64)

    @property
    def dtype(self):
        return self._data.dtype

