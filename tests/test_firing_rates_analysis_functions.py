import unittest

import numpy as np

from anaspike.firing_rates import (FiringRates,
                                   mean,
                                   std,
                                   construct_histogram)
from anaspike.dataclasses.coords2d import Coords2D
from anaspike.dataclasses.bins import ContigBins1D
from anaspike.dataclasses.grid import RegularGrid1D
from anaspike.dataclasses.interval import Interval



class TestMean(unittest.TestCase):
    def test_constant_fr(self):
        fr = FiringRates(Coords2D(np.arange(10), np.arange(10)),
                                    np.ones(10))
        expected_result = 1
        self.assertAlmostEqual(expected_result, mean(fr))

    def test_ascending_fr(self):
        start = 0.
        stop = 10.
        n = 10
        fr = FiringRates(Coords2D(np.arange(n), np.arange(n)),
                                    np.linspace(start, stop, n, dtype=np.float64))
        expected_result = (start + stop) / 2
        self.assertAlmostEqual(expected_result, mean(fr))


class TestStd(unittest.TestCase):
    def test_constant_fr(self):
        n = 10
        fr = FiringRates(Coords2D(np.arange(n), np.arange(n)),
                                    np.ones(n))
        expected_result = 0
        self.assertAlmostEqual(expected_result, std(fr))

    def test_ascending_fr(self):
        start = 0.
        stop = 10.
        n = 10
        fr = FiringRates(Coords2D(np.arange(n), np.arange(n)),
                                    np.linspace(start, stop, n, dtype=np.float64))
        step_size = (stop - start) / (n - 1)
        expected_result = step_size * np.sqrt((n**2 - 1) / 12)
        self.assertAlmostEqual(expected_result, std(fr))


class TestHistogramConstantFiringRate(unittest.TestCase):
    def setUp(self):
        n = 10
        self.fr = FiringRates(Coords2D(np.arange(n), np.arange(n)),
                                         np.ones(n))
        self.interval = Interval(0, 2)
        self.n_bins = 2

        self.expected_edges = np.array([0, 1, 2])
        self.expected_counts = np.array([0, 10])

    def test(self):
        bins = ContigBins1D[RegularGrid1D].with_median_labels(
                RegularGrid1D.from_interval_given_n(self.interval,
                                                    self.n_bins + 1,
                                                    endpoint=True))
        histogram = construct_histogram(self.fr, bins)

        np.testing.assert_array_equal(self.expected_edges, histogram.edges)
        np.testing.assert_array_equal(self.expected_counts, histogram.counts)

class TestHistogramAscendingFiringRate(unittest.TestCase):
    def setUp(self):
        self.start = 0.
        self.stop = 10.
        self.n_fr = 10
        self.fr = FiringRates(Coords2D(np.arange(self.n_fr),
                                                  np.arange(self.n_fr)),
                                         np.linspace(self.start, self.stop, self.n_fr, dtype=np.float64))

        self.interval = Interval(0, 10)
        self.n_bins = 5

        self.expected_edges = np.array([0, 2, 4, 6, 8, 10])
        self.expected_counts = np.array([2, 2, 2, 2, 2])

    def test(self):
        bins = ContigBins1D[RegularGrid1D].with_median_labels(
                RegularGrid1D.from_interval_given_n(self.interval,
                                                    self.n_bins +1,
                                                    endpoint=True))
        histogram = construct_histogram(self.fr, bins)

        np.testing.assert_array_equal(self.expected_edges, histogram.edges)
        np.testing.assert_array_equal(self.expected_counts, histogram.counts)


class BinSpatiallyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from anaspike.dataclasses.bins import ContigBins2D
        cls.ContigBins2D = ContigBins2D
        from anaspike.dataclasses.grid import RegularGrid2D, RegularGrid1D
        cls.RegularGrid2D = RegularGrid2D
        cls.RegularGrid1D = RegularGrid1D
        from anaspike.dataclasses.interval import Interval
        cls.Interval = Interval
        from anaspike.dataclasses.coords2d import Coords2D
        cls.Coords2D = Coords2D


class TestBinSpatiallySuccessful(BinSpatiallyTestCase):
    def setUp(self):
        import numpy as np
        from anaspike.dataclasses.bins import ContigBins2D
        from anaspike.dataclasses.grid import RegularGrid2D, RegularGrid1D
        from anaspike.dataclasses.interval import Interval
        self.fr = FiringRates(
                self.Coords2D(x=np.array([0.1, 0.4, 0.6, 0.8, 0.2, 0.3, 0.5]),
                              y=np.array([1.5, 0.7, 0.2, 0.9, 1.9, 1.4, 0.8])),
                np.array([1, 2, 3, 4, 5, 6, 7]))
        bin_grid = RegularGrid2D(RegularGrid1D.from_interval_given_n(Interval(0., 1.),
                                                          n=3,
                                                          endpoint=True),
                                 RegularGrid1D.from_interval_given_n(Interval(0., 2.),
                                                                     n=4,
                                                                     endpoint=True))
        self.bins = ContigBins2D[RegularGrid2D].with_median_labels(bin_grid)
        self.expected_firing_rates = np.array([[np.nan, 3.],
                                               [2.    , 11./2],
                                               [12./3 , np.nan]]).T
                                               
    def test(self):
        from anaspike.firing_rates import bin_spatially
        binned_fr = bin_spatially(self.fr, self.bins)
        np.testing.assert_array_almost_equal(binned_fr.elements, self.expected_firing_rates)

