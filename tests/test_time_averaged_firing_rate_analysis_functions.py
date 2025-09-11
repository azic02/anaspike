import unittest

import numpy as np

from anaspike.analysis.time_averaged_firing_rate import (TimeAveragedFiringRate,
                                                         mean,
                                                         std,
                                                         construct_histogram)



class TestMean(unittest.TestCase):
    def test_constant_fr(self):
        fr = TimeAveragedFiringRate(np.ones(10))
        expected_result = 1
        self.assertAlmostEqual(expected_result, mean(fr))

    def test_ascending_fr(self):
        start = 0.
        stop = 10.
        n = 10
        fr = TimeAveragedFiringRate(np.linspace(start, stop, n, dtype=np.float64))
        expected_result = (start + stop) / 2
        self.assertAlmostEqual(expected_result, mean(fr))


class TestStd(unittest.TestCase):
    def test_constant_fr(self):
        fr = TimeAveragedFiringRate(np.ones(10))
        expected_result = 0
        self.assertAlmostEqual(expected_result, std(fr))

    def test_ascending_fr(self):
        start = 0.
        stop = 10.
        n = 10
        fr = TimeAveragedFiringRate(np.linspace(start, stop, n, dtype=np.float64))
        step_size = (stop - start) / (n - 1)
        expected_result = step_size * np.sqrt((n**2 - 1) / 12)
        self.assertAlmostEqual(expected_result, std(fr))


class HistogramTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from anaspike.dataclasses.histogram import EquiBins
        cls.EquiBins = EquiBins
        from anaspike.dataclasses.interval import Interval
        cls.Interval = Interval

class TestHistogramConstantFiringRate(HistogramTestCase):
    def setUp(self):
        self.fr = TimeAveragedFiringRate(np.ones(10))
        self.interval = self.Interval(0, 2)
        self.n_bins = 2

        self.expected_bin_edges = np.array([0, 1, 2])
        self.expected_counts = np.array([0, 10])

    def test(self):
        bins = self.EquiBins.from_interval_with_median_values(self.interval, self.n_bins)
        histogram = construct_histogram(self.fr, bins)

        np.testing.assert_array_equal(self.expected_bin_edges, histogram.bin_edges)
        np.testing.assert_array_equal(self.expected_counts, histogram.counts)

class TestHistogramAscendingFiringRate(HistogramTestCase):
    def setUp(self):
        self.start = 0.
        self.stop = 10.
        self.n_fr = 10
        self.fr = TimeAveragedFiringRate(np.linspace(self.start, self.stop, self.n_fr, dtype=np.float64))

        self.interval = self.Interval(0, 10)
        self.n_bins = 5

        self.expected_bin_edges = np.array([0, 2, 4, 6, 8, 10])
        self.expected_counts = np.array([2, 2, 2, 2, 2])

    def test(self):
        bins = self.EquiBins.from_interval_with_median_values(self.interval, self.n_bins)
        histogram = construct_histogram(self.fr, bins)

        np.testing.assert_array_equal(self.expected_bin_edges, histogram.bin_edges)
        np.testing.assert_array_equal(self.expected_counts, histogram.counts)

