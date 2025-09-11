import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from anaspike.dataclasses import SpikeTrainArray



class TestSpikeTrainArray(unittest.TestCase):
    def test_different_size_spike_train_array_dtype(self):
        spike_trains = SpikeTrainArray([np.array([1., 2., 3.]),
                                        np.array([4., 5.])])
        self.assertEqual(spike_trains.dtype, np.dtype('O'))
        for train in spike_trains:
            self.assertIsInstance(train, np.ndarray)
            self.assertEqual(train.dtype, np.float64)

    def test_same_size_spike_train_array_dtype(self):
        spike_trains = SpikeTrainArray([np.array([1., 2., 3.]),
                                        np.array([4., 5., 6.])])
        self.assertEqual(spike_trains.dtype, np.dtype('O'))
        for train in spike_trains:
            self.assertIsInstance(train, np.ndarray)

    def test_single_spike_train_array_dtype(self):
        spike_trains = SpikeTrainArray([np.array([1., 2., 3.])])
        self.assertEqual(spike_trains.dtype, np.dtype('O'))
        self.assertIsInstance(spike_trains[0], np.ndarray)
        self.assertEqual(spike_trains[0].dtype, np.float64)

    def test_hdf5_conversion(self):
        import h5py
        spike_trains = SpikeTrainArray([np.array([1., 2., 3.]),
                                        np.array([4., 5.])])

        with h5py.File('test_spike_trains.h5', 'w') as f:
            spike_trains.to_hdf5(f, 'spike_trains')
        with h5py.File('test_spike_trains.h5', 'r') as f:
            loaded_trains = SpikeTrainArray.from_hdf5(f['spike_trains'])

        self.assertEqual(len(loaded_trains), len(spike_trains))
        self.assertEqual(loaded_trains.dtype, np.dtype('O'))
        for original, loaded in zip(spike_trains, loaded_trains):
            self.assertIsInstance(loaded, np.ndarray)
            self.assertEqual(loaded.dtype, np.float64)
            np.testing.assert_array_equal(original, loaded)


class TestSpikeTrainArrayFromNest(unittest.TestCase):
    @patch('anaspike.dataclasses.nest_devices.PopulationData')
    @patch('anaspike.dataclasses.nest_devices.SpikeRecorderData')
    def test_from_nest(self, MockSpikeRecorderData: MagicMock, MockPopulationData: MagicMock):
        mock_pop = MockPopulationData()
        mock_pop.ids = np.array([0, 1, 2], dtype=np.int64)

        mock_sr = MockSpikeRecorderData()
        mock_sr.times = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
        mock_sr.senders = np.array([0, 1, 0, 1, 2], dtype=np.int64)

        expected_spike_trains = [
            np.array([0.1, 0.3], dtype=np.float64),
            np.array([0.2, 0.4], dtype=np.float64),
            np.array([0.5], dtype=np.float64)
        ]

        spike_trains = SpikeTrainArray.from_nest(mock_pop, mock_sr)

        self.assertEqual(len(spike_trains), len(expected_spike_trains))
        for sr, exp_sr in zip(spike_trains, expected_spike_trains):
            np.testing.assert_array_almost_equal(exp_sr, sr)



if __name__ == "__main__":
    unittest.main(verbosity=2)

