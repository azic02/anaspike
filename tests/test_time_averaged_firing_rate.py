import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from anaspike.analysis.time_averaged_firing_rate import TimeAveragedFiringRate



class TestInit(unittest.TestCase):
    def test_valid_init(self):
        fr = TimeAveragedFiringRate(firing_rates = np.array([0.0, 1.0, 2.0]))
        self.assertIsInstance(fr, TimeAveragedFiringRate)
        np.testing.assert_array_almost_equal(fr, np.array([0.0, 1.0, 2.0]))

    def test_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            TimeAveragedFiringRate(firing_rates = np.array([[0.0, 1.0],
                                                            [2.0, 3.0]]))


class TestFromSpikeTrains(unittest.TestCase):
    @patch("anaspike.dataclasses.interval.Interval")
    @patch("anaspike.dataclasses.spike_train.SpikeTrainArray")
    def test_from_spike_trains(self, mock_spike_trains: MagicMock, mock_interval: MagicMock):
        mock_spike_trains.__iter__.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5],
            [0.6]
        ]
        mock_interval.contains.side_effect = lambda st: [True for t in st if 0.15 <= t <= 0.45]
        mock_interval.width = 0.3

        fr = TimeAveragedFiringRate.from_spike_trains(
            spike_trains=mock_spike_trains,
            time_window=mock_interval,
            time_unit=1.e-3
        )

        expected_rates = [2 / (0.3 * 1.e-3), 1 / (0.3 * 1.e-3), 0 / (0.3 * 1.e-3)]
        np.testing.assert_array_almost_equal(fr, expected_rates)


class TestMemberFuncs(unittest.TestCase):
    def setUp(self):
        self.fr = TimeAveragedFiringRate(firing_rates = np.array([0.0, 1.0, 2.0]))

    def test_as_nparray(self):
        self.assertIsInstance(self.fr.as_nparray, np.ndarray)
        self.assertEqual(self.fr.as_nparray.dtype, np.float64)
        np.testing.assert_array_almost_equal(self.fr.as_nparray, np.array([0.0, 1.0, 2.0]))

    def test_array(self):
        arr = np.array(self.fr)
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.dtype, np.float64)
        np.testing.assert_array_almost_equal(arr, np.array([0.0, 1.0, 2.0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)

