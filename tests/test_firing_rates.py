import unittest

import numpy as np

from anaspike.firing_rates import FiringRates



class TestInit(unittest.TestCase):
    def setUp(self):
        from anaspike.dataclasses.coords2d import Coords2D
        self.coords = Coords2D(x=[0.0, 1.0, 2.0], y=[0.0, 1.0, 2.0])
    def test_valid_init(self):
        fr = FiringRates(self.coords, firing_rates=np.array([0.0, 1.0, 2.0]))
        self.assertIsInstance(fr, FiringRates)
        np.testing.assert_array_almost_equal(fr.values, np.array([0.0, 1.0, 2.0]))

    def test_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            FiringRates(self.coords,
                                   firing_rates=np.array([[0.0, 1.0],
                                                          [2.0, 3.0]]))


class TestFromSpikeTrains(unittest.TestCase):
    def setUp(self):
        from anaspike.dataclasses.coords2d import Coords2D

        self.coords = Coords2D(x=[0.0, 1.0, 2.0], y=[0.0, 1.0, 2.0])
    def test_from_spike_trains(self):
        from anaspike.dataclasses.interval import Interval
        from anaspike.spike_trains import SpikeTrains

        spike_trains = SpikeTrains(self.coords, [
            np.array([0.1, 0.2, 0.3], dtype=np.float64),
            np.array([0.4, 0.5], dtype=np.float64),
            np.array([0.6], dtype=np.float64)
        ])

        interval = Interval(0.15, 0.45)

        fr = FiringRates.from_spike_trains(
            spike_trains=spike_trains,
            time_window=interval,
            time_unit=1.e-3
        )

        expected_rates = [2 / (0.3 * 1.e-3), 1 / (0.3 * 1.e-3), 0 / (0.3 * 1.e-3)]
        np.testing.assert_array_almost_equal(fr.values, expected_rates)


class TestHdf5Conversion(unittest.TestCase):
    def setUp(self):
        from pathlib import Path
        from anaspike.dataclasses.coords2d import Coords2D
        self.file_path = Path('test_ta_fr.h5')
        self.fr = FiringRates(
                coords=Coords2D(x=[0.0, 1.0, 2.0], y=[0.0, 1.0, 2.0]),
                firing_rates=np.array([0.0, 1.0, 2.0]))

    def test_to_hdf5_and_from_hdf5(self):
        import h5py
        with h5py.File(self.file_path, 'w') as f_out:
            self.fr.to_hdf5(f_out, 'firing_rates')
        with h5py.File(self.file_path, 'r') as f_in:
            fr_loaded = FiringRates.from_hdf5(f_in['firing_rates'])
        np.testing.assert_array_almost_equal(fr_loaded.values, self.fr.values)
        np.testing.assert_array_almost_equal(fr_loaded.coords.x, self.fr.coords.x)
        np.testing.assert_array_almost_equal(fr_loaded.coords.y, self.fr.coords.y)

    def cleanUp(self):
        import os
        if os.path.exists(self.file_path):
            os.remove(self.file_path)



if __name__ == "__main__":
    unittest.main(verbosity=2)

