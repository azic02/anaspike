import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from anaspike.analysis.time_averaged_firing_rate import TimeAveragedFiringRate



class TestInit(unittest.TestCase):
    def setUp(self):
        from anaspike.dataclasses.coords2d import Coords2D
        self.coords = Coords2D(x=[0.0, 1.0, 2.0], y=[0.0, 1.0, 2.0])
    def test_valid_init(self):
        fr = TimeAveragedFiringRate(self.coords, firing_rates=np.array([0.0, 1.0, 2.0]))
        self.assertIsInstance(fr, TimeAveragedFiringRate)
        np.testing.assert_array_almost_equal(fr, np.array([0.0, 1.0, 2.0]))

    def test_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            TimeAveragedFiringRate(self.coords,
                                   firing_rates=np.array([[0.0, 1.0],
                                                          [2.0, 3.0]]))


class TestFromSpikeTrains(unittest.TestCase):
    def setUp(self):
        from anaspike.dataclasses.coords2d import Coords2D
        self.coords = Coords2D(x=[0.0, 1.0, 2.0], y=[0.0, 1.0, 2.0])
    @patch("anaspike.dataclasses.interval.Interval")
    @patch("anaspike.analysis.spike_trains.SpikeTrainArray")
    def test_from_spike_trains(self, mock_spike_trains: MagicMock, mock_interval: MagicMock):
        mock_spike_trains.__iter__.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5],
            [0.6]
        ]
        mock_interval.contains.side_effect = lambda st: [True for t in st if 0.15 <= t <= 0.45]
        mock_interval.width = 0.3

        fr = TimeAveragedFiringRate.from_spike_trains(
            coords=self.coords,
            spike_trains=mock_spike_trains,
            time_window=mock_interval,
            time_unit=1.e-3
        )

        expected_rates = [2 / (0.3 * 1.e-3), 1 / (0.3 * 1.e-3), 0 / (0.3 * 1.e-3)]
        np.testing.assert_array_almost_equal(fr, expected_rates)


class TestHdf5Conversion(unittest.TestCase):
    def setUp(self):
        from pathlib import Path
        from anaspike.dataclasses.coords2d import Coords2D
        self.file_path = Path('test_ta_fr.h5')
        self.fr = TimeAveragedFiringRate(
                coords=Coords2D(x=[0.0, 1.0, 2.0], y=[0.0, 1.0, 2.0]),
                firing_rates=np.array([0.0, 1.0, 2.0]))

    def test_to_hdf5_and_from_hdf5(self):
        import h5py
        with h5py.File(self.file_path, 'w') as f_out:
            self.fr.to_hdf5(f_out, 'time_averaged_firing_rate')
        with h5py.File(self.file_path, 'r') as f_in:
            fr_loaded = TimeAveragedFiringRate.from_hdf5(f_in['time_averaged_firing_rate'])
        np.testing.assert_array_almost_equal(fr_loaded, self.fr)
        np.testing.assert_array_almost_equal(fr_loaded.coords.x, self.fr.coords.x)
        np.testing.assert_array_almost_equal(fr_loaded.coords.y, self.fr.coords.y)

    def cleanUp(self):
        import os
        if os.path.exists(self.file_path):
            os.remove(self.file_path)



if __name__ == "__main__":
    unittest.main(verbosity=2)

