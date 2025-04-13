from dataclasses import dataclass
from typing import Union

import numpy as np
from numpy.typing import NDArray
import h5py

from ...functions._helpers import validate_one_dimensional, validate_same_length



@dataclass
class PopulationData:
    ids: NDArray[np.int64]
    x_pos: NDArray[np.float64]
    y_pos: NDArray[np.float64]

    @classmethod
    def from_hdf5(cls, entity: Union[h5py.Group, h5py.File]) -> 'PopulationData':
        return cls(np.array(entity['ids'], dtype=np.int64),
                   np.array(entity['x_pos'], dtype=np.float64),
                   np.array(entity['y_pos'], dtype=np.float64),
                   )

    def to_hdf5(self, entity: Union[h5py.Group, h5py.File]) -> None:
        entity.create_dataset('ids', data=self.ids)
        entity.create_dataset('x_pos', data=self.x_pos)
        entity.create_dataset('y_pos', data=self.y_pos)

    def __post_init__(self):
        validate_one_dimensional(self.ids)
        validate_one_dimensional(self.x_pos)
        validate_one_dimensional(self.y_pos)
        validate_same_length([self.ids, self.x_pos, self.y_pos])

    def __len__(self):
        return len(self.ids)

    def __add__(self, other: 'PopulationData') -> 'PopulationData':
        return PopulationData(ids=np.hstack([self.ids, other.ids]),
                          x_pos=np.hstack([self.x_pos, other.x_pos]),
                          y_pos=np.hstack([self.y_pos, other.y_pos]),
                          )

    def __getitem__(self, key) -> Union['NeuronData','PopulationData']:
        return (NeuronData(np.array([self.ids[key]], dtype=np.int64),
                           np.array([self.x_pos[key]]),
                           np.array([self.y_pos[key]]),
                           )
                if type(key) == int
                else PopulationData(self.ids[key], self.x_pos[key], self.y_pos[key])
                )


@dataclass
class NeuronData(PopulationData):
    def __post_init__(self):
        super().__post_init__()
        if len(self) != 1:
            raise ValueError("NeuronData must contain exactly one neuron.")

