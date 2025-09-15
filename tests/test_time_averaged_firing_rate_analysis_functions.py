import unittest

import numpy as np

from anaspike.analysis.time_averaged_firing_rate import (TimeAveragedFiringRate,
                                                         mean,
                                                         std,
                                                         construct_histogram)
from anaspike.dataclasses.coords2d import Coords2D



class TestMean(unittest.TestCase):
    def test_constant_fr(self):
        fr = TimeAveragedFiringRate(Coords2D(np.arange(10), np.arange(10)),
                                    np.ones(10))
        expected_result = 1
        self.assertAlmostEqual(expected_result, mean(fr))

    def test_ascending_fr(self):
        start = 0.
        stop = 10.
        n = 10
        fr = TimeAveragedFiringRate(Coords2D(np.arange(n), np.arange(n)),
                                    np.linspace(start, stop, n, dtype=np.float64))
        expected_result = (start + stop) / 2
        self.assertAlmostEqual(expected_result, mean(fr))


class TestStd(unittest.TestCase):
    def test_constant_fr(self):
        n = 10
        fr = TimeAveragedFiringRate(Coords2D(np.arange(n), np.arange(n)),
                                    np.ones(n))
        expected_result = 0
        self.assertAlmostEqual(expected_result, std(fr))

    def test_ascending_fr(self):
        start = 0.
        stop = 10.
        n = 10
        fr = TimeAveragedFiringRate(Coords2D(np.arange(n), np.arange(n)),
                                    np.linspace(start, stop, n, dtype=np.float64))
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
        n = 10
        self.fr = TimeAveragedFiringRate(Coords2D(np.arange(n), np.arange(n)),
                                         np.ones(n))
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
        self.fr = TimeAveragedFiringRate(Coords2D(np.arange(self.n_fr),
                                                  np.arange(self.n_fr)),
                                         np.linspace(self.start, self.stop, self.n_fr, dtype=np.float64))

        self.interval = self.Interval(0, 10)
        self.n_bins = 5

        self.expected_bin_edges = np.array([0, 2, 4, 6, 8, 10])
        self.expected_counts = np.array([2, 2, 2, 2, 2])

    def test(self):
        bins = self.EquiBins.from_interval_with_median_values(self.interval, self.n_bins)
        histogram = construct_histogram(self.fr, bins)

        np.testing.assert_array_equal(self.expected_bin_edges, histogram.bin_edges)
        np.testing.assert_array_equal(self.expected_counts, histogram.counts)


class BinSpatiallyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from anaspike.dataclasses.contig_bins_2d import ContigBins2D
        cls.ContigBins2D = ContigBins2D
        from anaspike.dataclasses.histogram import EquiBins
        cls.EquiBins = EquiBins
        from anaspike.dataclasses.interval import Interval
        cls.Interval = Interval
        from anaspike.dataclasses.coords2d import Coords2D
        cls.Coords2D = Coords2D


class TestBinSpatiallySuccessful(BinSpatiallyTestCase):
    def setUp(self):
        import numpy as np
        self.fr = TimeAveragedFiringRate(
                self.Coords2D(x=np.array([0.1, 0.4, 0.6, 0.8, 0.2, 0.3, 0.5]),
                              y=np.array([1.5, 0.7, 0.2, 0.9, 1.9, 1.4, 0.8])),
                np.array([1, 2, 3, 4, 5, 6, 7]))
        self.bins = self.ContigBins2D(self.EquiBins.from_interval_with_median_values(self.Interval(0, 1), n=2),
                                      self.EquiBins.from_interval_with_median_values(self.Interval(0, 2), n=3))
        self.expected_firing_rates = np.array([[np.nan, 3.],
                                               [2.    , 11./2],
                                               [12./3 , np.nan]]).T
                                               
    def test(self):
        from anaspike.analysis.time_averaged_firing_rate import bin_spatially
        binned_fr = bin_spatially(self.fr, self.bins)
        np.testing.assert_array_almost_equal(binned_fr.elements, self.expected_firing_rates)

