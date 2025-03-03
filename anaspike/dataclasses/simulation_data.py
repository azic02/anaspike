from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import h5py

from .nest_devices.population_data import PopulationData
from .nest_devices.spike_recorder_data import SpikeRecorderData



@dataclass
class SimulationData:
    populations: Dict[str, PopulationData]
    spike_recorder: SpikeRecorderData


    @classmethod
    def load(cls, path: Path):
        with h5py.File(path, 'r') as f:
            return cls(populations={key.decode('utf-8'): PopulationData.from_hdf5(f['populations'][key]) for key in f['populations']['names'][:]},
                       spike_recorder=SpikeRecorderData.from_hdf5(f['spike_recorder']),
                       )


    def save(self, path: Path) -> None:
        with h5py.File(path, 'w') as f:
            pops_group = f.create_group('populations')
            pops_group.create_dataset('names', data=list(self.populations.keys()))
            for name, pop in self.populations.items():
                pop_group = pops_group.create_group(name)
                pop.to_hdf5(pop_group)

            self.spike_recorder.to_hdf5(f.create_group('spike_recorder'))

