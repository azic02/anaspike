import unittest

import numpy as np

from anaspike.dataclasses import SpikeTrainArray



class TestSpikeTrainArray(unittest.TestCase):
    def test_different_size_spike_train_array_dtype(self):
        spike_trains = SpikeTrainArray([np.array([1., 2., 3.]),
                                        np.array([4., 5.])])
        self.assertIs(spike_trains.dtype, np.dtype('O'))

    def test_same_size_spike_train_array_dtype(self):
        spike_trains = SpikeTrainArray([np.array([1., 2., 3.]),
                                        np.array([4., 5., 6.])])
        self.assertIs(spike_trains.dtype, np.dtype('O'))

    def test_single_spike_train_array_dtype(self):
        spike_trains = SpikeTrainArray([np.array([1., 2., 3.])])
        self.assertIs(spike_trains.dtype, np.dtype('O'))


if __name__ == "__main__":
    unittest.main(verbosity=2)

