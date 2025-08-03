import unittest

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
        #self.assertEqual(spike_trains[0].dtype, np.float64)

    def test_hdf5_conversion(self):
        import h5py
        spike_trains = SpikeTrainArray([np.array([1., 2., 3.]),
                                        np.array([4., 5.])])

        with h5py.File('test_spike_trains.h5', 'w') as f:
            f.create_dataset('spike_trains', data=spike_trains,
                             dtype=h5py.vlen_dtype(np.float64))

        with h5py.File('test_spike_trains.h5', 'r') as f:
            loaded_trains = SpikeTrainArray(f['spike_trains'])

        self.assertEqual(len(loaded_trains), len(spike_trains))
        self.assertEqual(loaded_trains.dtype, np.dtype('O'))
        for original, loaded in zip(spike_trains, loaded_trains):
            self.assertIsInstance(loaded, np.ndarray)
            self.assertEqual(loaded.dtype, np.float64)
            np.testing.assert_array_equal(original, loaded)



if __name__ == "__main__":
    unittest.main(verbosity=2)

