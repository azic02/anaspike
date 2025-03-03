import unittest

import numpy as np

from anaspike.dataclasses.nest_devices import SpikeRecorderData



class TestAdd(unittest.TestCase):
    def test_add(self):
        spike_recorder_data_1 = SpikeRecorderData(
            senders=np.array([0, 1, 0, 1, 2], dtype=np.int64),
            times=np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
        )

        spike_recorder_data_2 = SpikeRecorderData(
            senders=np.array([3, 4, 3, 4, 5], dtype=np.int64),
            times=np.array([0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float64)
        )

        spike_recorder_data_sum = spike_recorder_data_1 + spike_recorder_data_2

        expected_spike_recorder_data_sum = SpikeRecorderData(
            senders=np.array([0, 1, 0, 1, 2, 3, 4, 3, 4, 5], dtype=np.int64),
            times=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float64)
        )

        np.testing.assert_array_almost_equal(spike_recorder_data_sum.senders, expected_spike_recorder_data_sum.senders)
        np.testing.assert_array_almost_equal(spike_recorder_data_sum.times, expected_spike_recorder_data_sum.times)


class TestGetSpikeTrains(unittest.TestCase):
    def test_get_spike_trains(self):
        spike_recorder_data = SpikeRecorderData(
            senders=np.array([0, 1, 0, 1, 2], dtype=np.int64),
            times=np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
        )

        spike_trains = spike_recorder_data.get_spike_trains([0, 1, 2])

        expected_spike_trains = [
            np.array([0.1, 0.3], dtype=np.float64),
            np.array([0.2, 0.4], dtype=np.float64),
            np.array([0.5], dtype=np.float64)
        ]

        for res, exp in zip(spike_trains, expected_spike_trains):
            np.testing.assert_array_almost_equal(res, exp)


if __name__ == '__main__':
    unittest.main(verbosity=2)

